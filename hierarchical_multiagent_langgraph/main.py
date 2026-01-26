"""Hierarchical Multiagent LangGraph ROS2 Node.

This module implements a hierarchical multi-agent system using LangGraph
and ROS2. It manages a supervisor agent that coordinates multiple specialized
single-purpose agents (SPAs) for complex task execution. The supervisor
decomposes high-level user queries into sub-tasks and delegates them to
appropriate agents, then synthesizes their responses.

Main Components:
    - HierarchicalMultiagent: Main ROS2 node managing the supervisor and agents.
    - SupervisorManager: Orchestrates the agent hierarchy and LangGraph workflow.
    - Agent execution threads: Each agent runs in its own event loop for isolation.
"""

from hierarchical_multiagent_langgraph.supervisor import (
    InputState,
    SupervisorManager
)
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from llm_interactions_msgs.srv import CallAgent


import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


class HierarchicalMultiagent(LangGraphRosBase):
    """
    ROS2 node orchestrating a hierarchical multi-agent system using LangGraph.

    This node implements a supervisor-based architecture where a single supervisor
    agent decomposes complex user queries into subtasks and coordinates multiple
    single-purpose agents (SPAs) for parallel execution. The supervisor uses LLM-based
    reasoning to intelligently distribute work and synthesize final responses from
    multiple agent outputs.

    Architecture:
        - Supervisor Agent: Analyzes user queries and creates/manages SPAs via tools.
        - Single-Purpose Agents: Each executes a focused subtask assigned by supervisor.
        - Event Loop Management: Each agent runs in isolated async context for concurrency.
        - ROS2 Service Interface: Accepts queries via CallAgent service.

    Execution Flow:
        1. ROS2 service receives user query (CallAgent request).
        2. Query forwarded to supervisor's LangGraph workflow.
        3. Supervisor decides to create/delete agents based on LLM reasoning.
        4. Created agents queued in pending_agents_list via supervisor tools.
        5. Supervisor's consume_pending_agents() executes agents in background threads.
        6. Each agent runs with isolated asyncio event loop for non-blocking concurrency.
        7. Results collected asynchronously from completed agents.
        8. Supervisor synthesizes final response from agent results.
        9. Response returned via ROS2 service callback.

    Thread Safety:
        Agent lists are protected by agent_lists_lock. Agent execution runs in
        separate threads with independent event loops to prevent blocking the
        ROS2 executor.

    Attributes:
        supervisor_manager (SupervisorManager): Manages supervisor agent and LangGraph workflow.
        agent_srv (rclpy.node.Service): ROS2 service endpoint for receiving queries.
        spa_params (dict): Configuration passed to all SinglePurposeAgent instances.
        loop (asyncio.AbstractEventLoop): Main event loop for supervisor execution.
        max_steps (int): Maximum LangGraph steps for supervisor and agents.

    Raises:
        No explicit raises, but initialization failures in supervisor setup may
        propagate from SupervisorManager or LangGraphBase parent classes.
    """

    def __init__(self):
        """
        Initialize the Hierarchical Multiagent ROS2 node.

        Performs complete initialization sequence:
        1. Calls parent class (LangGraphRosBase) initializer.
        2. Retrieves SPA configuration parameters.
        3. Creates SupervisorManager instance.
        4. Retrieves tools available to supervisor from Ollama.
        5. Builds the supervisor's LangGraph workflow.
        6. Creates ROS2 service endpoint for receiving user queries.
        7. Sets up timer for periodic delegation to supervisor's agent consumption.

        After successful initialization, the node is ready to receive user queries
        via ROS2 service and manage the hierarchical multi-agent execution.

        Raises:
            Various exceptions from parent class or SupervisorManager if
            configuration loading, LLM connection, or LangGraph building fails.
        """
        # Call the base class initializer
        super().__init__()

        self.get_spa_params()

        # Create the subscriber to listen for user queries
        self._timer_group = ReentrantCallbackGroup()
        self._srv_group = MutuallyExclusiveCallbackGroup()
        self.agent_srv = self.create_service(
            srv_type=CallAgent,
            srv_name=self.service_name,
            callback=self.agent_callback,
            callback_group=self._srv_group
        )
        # Create timer to consume pending agents from the queue
        # Uses ReentrantCallbackGroup to allow concurrent execution
        self.agent_timer = self.create_timer(
            1.0,  # Timer period in seconds
            self._consume_pending_agents_timer_callback,
            callback_group=self._timer_group
        )

        # Initialize Ollama agent with retry logic
        self.initialize_ollama_with_retries(
            mcp_servers=self.mcp_servers,
            agent_params=self.agent_params,
            max_retries=5,
            retry_delay=2.0
        )
        # Initialize the Supervisor Manager
        self.supervisor_manager = SupervisorManager(
            logger=self.get_logger(),
            ollama_agent=self.ollama_agent,
            max_steps=self.max_steps,
            system_prompt_path=self.system_prompt_file,
            spa_params=self.spa_params
        )

        # Retrieve tools for Ollama agent
        self.loop.run_until_complete(
            self.supervisor_manager.ollama_agent.retrieve_tools(
                self.supervisor_manager.supervisor_tools
            ))

        # Build the LangGraph workflow
        self.build_graph()

        self.get_logger().info('Hierarchical Multiagent LangGraph Node has been started.')

    def _consume_pending_agents_timer_callback(self) -> None:
        """
        Timer callback: delegate agent consumption to supervisor.

        Called periodically (every 1.0 second) by ROS2 timer.

        Parameters:
            None: Uses supervisor_manager instance.

        Returns:
            None
        """
        self.supervisor_manager.consume_pending_agents()

    def build_graph(self) -> None:
        """
        Initialize and compile the supervisor's LangGraph workflow.

        Calls supervisor_manager.make_graph() asynchronously in the main event loop
        to construct the supervisor's state machine. The graph implements the
        hierarchical multi-agent coordination workflow with nodes for initial setup,
        task analysis, agent management (via tools), and response synthesis.

        Graph Components:
            - set_initial_messages: Initialize conversation state from user prompt
            - analyze_task: LLM supervisor analyzes query and decides on agents
            - route_on_tool_call: Route supervisor tool calls (create/delete agents)
            - finalize_conversation: Aggregate agent results into final response

        Compilation:
            - Converts workflow definition to compiled graph (LangGraph)
            - Prepares graph for invocation (ainvoke, invoke)
            - Enables checkpoint/memory persistence via configurable thread_id

        Error Handling:
            - Catches and logs exceptions from make_graph()
            - Re-raises exceptions to prevent node startup with incomplete graph
            - Ensures node initialization fails fast on graph building errors

        Parameters:
            None: Uses supervisor_manager and main event loop

        Returns:
            None: Compiled graph stored in supervisor_manager.graph

        Side Effects:
            - Populates supervisor_manager.graph with compiled LangGraph
            - Logs success/failure of graph compilation
            - May raise exceptions that propagate to __init__
        """
        # Initialize and compile the LangGraph workflow
        try:
            self.loop.run_until_complete(self.supervisor_manager.make_graph())
        except Exception as e:
            self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
            raise

        self.get_logger().info('SupervisorManager graph created successfully...')

    def _process_graph_async(self, input_state: InputState, thread_id: str) -> None:
        """
        Asynchronously invoke supervisor's graph without blocking caller.

        Fire-and-forget execution: submits supervisor task and returns immediately.
        Supervisor operates concurrently with other ROS2 callbacks. Results printed
        to console after completion. Used when ROS2 caller doesn't need response
        (response_needed=False in service call).

        Execution Model:
            - Invokes graph.ainvoke() but doesn't wait for completion
            - Supervisor runs in main event loop background
            - Caller returns immediately with acknowledgment
            - Results logged/printed after supervisor finishes

        Use Cases:
            - Fire-and-forget queries: "turn on the light" (no response needed)
            - Background tasks: data processing, monitoring without user wait
            - Reduce latency: ROS2 callback can return quickly to executor

        Limitation:
            - Caller receives no direct result (only asynchronous console output)
            - No error feedback to caller (errors logged only)
            - Task runs but completion not guaranteed to caller

        Parameters:
            input_state (InputState): Initial state with 'user_prompt'.
                Format: {'user_prompt': 'user query string'}
            thread_id (str): Checkpoint thread identifier for state persistence.

        Returns:
            None: Returns immediately; actual processing continues in background.

        Side Effects:
            - Submits supervisor task to event loop
            - Prints result to console (not returned to ROS2 caller)
            - May trigger agent creation/execution in background
            - Updates supervisor_manager agent lists during background execution

        Note:
            This implementation still blocks via run_until_complete().
            For true non-blocking behavior, use asyncio.create_task() instead,
            but this requires careful exception handling for background tasks.
        """
        config = {'configurable': {'thread_id': thread_id}}
        result = self.loop.run_until_complete(
            self.supervisor_manager.graph.ainvoke(input_state, config=config)
        )
        # print(f'Asynchronous processing result: {result}')

    def agent_callback(self, request, response):
        """
        ROS2 service callback: process user queries asynchronously.

        Implements CallAgent service endpoint. Routes incoming user queries to
        supervisor's LangGraph workflow for asynchronous processing. Submits query
        to background execution without waiting for completion.

        Request Fields:
            - query (str): User's natural language query/task
            - response_needed (bool): Deprecated. Currently ignored; all queries
                processed asynchronously for non-blocking ROS2 service behavior.

        Execution Flow:
            1. Extract query from request.query
            2. Log incoming query at DEBUG level
            3. Create InputState dictionary with user_prompt key
            4. Set fixed thread_id='supervisor' for checkpoint context
            5. Call _process_graph_async(input_state, thread_id)
            6. Return immediately with success message
            7. Supervisor processes graph in background (may create agents, execute tools)
            8. Results logged/printed but not returned to caller

        Thread ID Management:
            - Uses fixed thread_id='supervisor' for all service calls
            - Maintains supervisor checkpoint context across multiple calls
            - Enables checkpoint persistence and state recovery from ROS2 node lifetime

        Response Population:
            - response.agent_response set to fixed string: 'Query submitted for processing'
            - Indicates query was accepted and queued, not that processing completed
            - Client should not rely on response_field for task results

        Logging:
            - Logs incoming query at DEBUG level
            - Does not log processing time (asynchronous execution)
            - Results printed to logger by supervisor in background

        Parameters:
            request (CallAgent.Request): Service request with query string.
            response (CallAgent.Response): Service response object to populate.

        Returns:
            CallAgent.Response: Response with fixed agent_response string.

        Side Effects:
            - Queues supervisor graph processing in event loop
            - Does not block ROS2 executor
            - Eventually modifies agent_lists (via background timer)
            - Logs query and status

        Note:
            Non-blocking service design allows ROS2 MultiThreadedExecutor to handle
            multiple concurrent requests without blocking on long-running graph
            execution. Graph processing happens in dedicated event loop thread managed
            by _agent_execution_timer_callback.
        """
        user_query = request.query

        self.get_logger().info(f'Received user query:\n{user_query}')

        # Use a fixed thread ID for supervisor to maintain context
        thread_id = 'supervisor'

        # Create input state with user prompt
        input_state: InputState = {'user_prompt': user_query}

        # Process graph
        self.get_logger().info('Processing query asynchronously...')
        self._process_graph_async(input_state, thread_id)
        response.agent_response = 'Query submitted for processing'

        self.get_logger().info('Query processing submitted; returning response to caller.')
        return response

    def get_spa_params(self) -> None:
        """
        Declare and retrieve ROS2 parameters for SinglePurposeAgent configuration.

        Loads SPA-specific configuration from ROS2 parameter server. Declares
        parameters with defaults, retrieves actual values (from launch files,
        param files, or defaults), and stores in self.spa_params dictionary.
        All values logged for verification.

        Parameters Loaded:
            - spa_mcp_servers (str, default='mcp.json'): Path to MCP servers config
            - spa_system_prompt_file (str, default='system_prompt.jinja'): Jinja2 template path
            - spa_template_type (str, default='qwen3'): LLM chat template type
            - spa_llm_model (str, default='qwen3:0.6b'): Model name for Ollama
            - spa_tool_call_pattern (str, default='<tool_call>(.*?)</tool_call>'): Regex pattern
            - spa_max_steps (int, default=5): Maximum LangGraph steps per agent

        Configuration Source Hierarchy:
            1. Launch file parameters (highest priority)
            2. Parameter file (.yaml) parameters
            3. Declared defaults (lowest priority)

        State Updates:
            - Copies self.agent_params (from parent class) to self.spa_params
            - Removes mcp_client if present (managed separately)
            - Adds/updates SPA-specific parameters

        Logging:
            - Logs each parameter name and value at INFO level
            - Provides verification that correct values loaded
            - Helps debug configuration issues

        Parameters:
            None: Uses ROS2 parameter server via self.get_parameter()

        Returns:
            None: Stores results in self.spa_params dictionary

        Side Effects:
            - Declares 6 ROS2 parameters (idempotent if already declared)
            - Populates self.spa_params with loaded values
            - Logs all parameter values to ROS2 logger

        Usage Context:
            - Called during HierarchicalMultiagent.__init__()
            - Before SupervisorManager creation
            - Provides config for all subsequent SPAs created by supervisor
        """
        # Initialize spa_params dictionary by copying agent_params
        self.spa_params = self.agent_params.copy()
        # Remove mcp_client from spa_params if it exists
        if 'mcp_client' in self.spa_params:
            del self.spa_params['mcp_client']

        # Declare and retrieve MCP servers parameter
        self.declare_parameter('spa_mcp_servers', 'mcp.json')
        self.spa_params['mcp_servers_config'] = self.get_parameter(
            'spa_mcp_servers').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_mcp_servers is set to: [{self.spa_params["mcp_servers_config"]}]')

        # Declare and retrieve system prompt template path parameter
        self.declare_parameter('spa_system_prompt_file', 'system_prompt.jinja')
        self.spa_params['system_prompt_file'] = self.get_parameter(
            'spa_system_prompt_file').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_system_prompt_file is set to: '
            f'[{self.spa_params["system_prompt_file"]}]')

        # Declare and retrieve model chat template file path parameter
        self.declare_parameter('spa_template_type', 'qwen3')
        self.spa_params['template_type'] = self.get_parameter(
            'spa_template_type').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_template_type is set to: [{self.spa_params["template_type"]}]')

        # Declare and retrieve model chat template file name parameter
        self.declare_parameter('spa_template_file', 'qwen3.jinja')
        self.spa_params['template_file'] = self.get_parameter(
            'spa_template_file').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_template_file is set to: [{self.spa_params["template_file"]}]')

        # Declare and retrieve LLM model name parameter
        self.declare_parameter('spa_llm_model', 'qwen3:0.6b')
        self.spa_params['model'] = self.get_parameter(
            'spa_llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_llm_model is set to: [{self.spa_params["model"]}]')
        # Declare tool call regex pattern to extract tool calls from LLM response
        self.declare_parameter('spa_tool_call_pattern', '<tool_call>(.*?)</tool_call>')
        self.spa_params['tool_call_pattern'] = self.get_parameter(
            'spa_tool_call_pattern').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_tool_call_pattern is set to: '
            f'[{self.spa_params["tool_call_pattern"]}]')

        # Declare and retrieve available tools for SPA agents
        self.declare_parameter('spa_available_tools', ['execute_behavior_tree'])
        self.spa_params['available_tools'] = self.get_parameter(
            'spa_available_tools').get_parameter_value().string_array_value
        self.get_logger().info(
            f'The parameter spa_available_tools is set to: '
            f'[{self.spa_params["available_tools"]}]')

        # Declare and retrieve LangGraph workflow parameters
        self.declare_parameter('spa_max_steps', 5)
        self.spa_params['max_steps'] = self.get_parameter(
            'spa_max_steps').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter spa_max_steps is set to: [{self.spa_params["max_steps"]}]')


def main(args=None) -> None:
    """
    Entry point: initialize ROS2 and run hierarchical multi-agent node.

    Performs complete ROS2 initialization and node lifecycle management:
    1. Initialize ROS2 context with rclpy.init()
    2. Create HierarchicalMultiagent node instance
    3. Create MultiThreadedExecutor for concurrent callback execution
    4. Add node to executor
    5. Spin executor until shutdown (Ctrl+C, external signal, or error)
    6. Cleanup handled by rclpy (shutdown on exception or exit)

    Execution Model:
        - MultiThreadedExecutor allows ROS2 callbacks to run concurrently
        - Supervisor timer callback (_agent_execution_timer_callback) runs periodically
        - Agent callback (agent_callback) handles service requests from multiple callers
        - Multiple agents can execute in parallel via timer callbacks

    Error Handling:
        - KeyboardInterrupt: User pressed Ctrl+C (expected shutdown)
        - ExternalShutdownException: ROS2 received shutdown signal (expected)
        - Exception: Any other error during node execution (unexpected)
        - All cases: Print shutdown reason and exit cleanly

    Node Lifecycle:
        1. Initializes HierarchicalMultiagent() which:
           - Creates supervisor manager
           - Builds LangGraph workflow
           - Registers ROS2 service
           - Starts timer for agent execution
        2. Executes until shutdown
        3. Cleanup handled by rclpy.shutdown()

    Parameters:
        args (list | None): Command-line arguments passed to rclpy.init().
            None uses sys.argv. Typical: ['--ros-args', '--log-level', 'info']

    Returns:
        None: Exits process via rclpy.shutdown() or exception

    Side Effects:
        - Initializes global ROS2 context (rclpy.init)
        - Creates HierarchicalMultiagent node
        - Registers ROS2 services and timers
        - Blocks until shutdown (run forever in normal operation)
        - Prints shutdown reason to console

    Usage:
        $ ros2 run hierarchical_multiagent_langgraph supervisor
        # Node runs until Ctrl+C or external shutdown signal
    """
    rclpy.init(args=args)

    try:
        # Create the agent node
        agent = HierarchicalMultiagent()

        # Use a MultiThreadedExecutor to allow concurrent callback execution
        executor = MultiThreadedExecutor()
        executor.add_node(agent)

        # Spin the node to process callbacks
        executor.spin()
    except (KeyboardInterrupt, Exception, ExternalShutdownException) as e:
        print(f'Shutting down agent node due to: {e}')


if __name__ == '__main__':
    main()
