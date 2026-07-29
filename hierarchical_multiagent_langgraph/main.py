"""
Hierarchical Multiagent LangGraph ROS2 Node.

This module implements a hierarchical multi-agent system using LangGraph
and ROS2. It manages a supervisor agent that coordinates multiple specialized
single-purpose agents (SPAs) for complex task execution. The supervisor
decomposes high-level user queries into sub-tasks and delegates them to
appropriate agents, then synthesizes their responses.

Main Components:
    - HierarchicalMultiagent: Main ROS2 node managing the supervisor and agents.
    - SupervisorManager: Orchestrates the agent hierarchy and LangGraph workflow.
    - Agent execution threads: Each agent runs in its own event loop, triggered
      by an event-driven consumer thread (no polling timer).
"""

from typing import Any

from hierarchical_multiagent_langgraph.supervisor import (
    InputState,
    SupervisorManager
)
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from llm_interactions_msgs.srv import CallAgent


import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

# Data-driven SPA parameter definitions: (ros_name, dict_key, default_value)
# The value type is inferred from the default to select the correct ROS2 accessor.
_SPA_PARAM_DEFS: list[tuple[str, str, Any]] = [
    ('spa_mcp_servers', 'mcp_servers_config', 'mcp.json'),
    ('spa_system_prompt_file', 'system_prompt_file', 'system_prompt.jinja'),
    ('spa_template_type', 'template_type', 'qwen3'),
    ('spa_template_file', 'template_file', 'qwen3.jinja'),
    ('spa_llm_model', 'model', 'qwen3:0.6b'),
    ('spa_tool_call_pattern', 'tool_call_pattern', '<tool_call>(.*?)</tool_call>'),
    ('spa_available_tools', 'available_tools', ['execute_behavior_tree']),
    ('spa_max_steps', 'max_steps', 5),
    ('spa_enable_thinking', 'think', False),
]

# Maps Python type -> ROS2 ParameterValue attribute name
_VALUE_ACCESSORS: dict[type, str] = {
    str: 'string_value',
    int: 'integer_value',
    float: 'double_value',
    bool: 'bool_value',
    list: 'string_array_value',
}


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
        5. AgentExecutor's consumer thread detects pending agents via event.
        6. Each agent runs in a dedicated thread with its own asyncio event loop.
        7. Results collected asynchronously from completed agents.
        8. Supervisor synthesizes final response from agent results.
        9. Response returned via ROS2 service callback.

    Thread Safety:
        Agent lists are protected by agent_lists_lock. Agent execution runs in
        separate threads with independent event loops to prevent blocking the
        ROS2 executor. The consumer thread uses threading.Event for zero-latency
        wake-up when new agents are enqueued.

    Attributes
    ----------
    supervisor_manager : SupervisorManager
        Manages supervisor agent and LangGraph workflow.
    agent_srv : rclpy.node.Service
        ROS2 service endpoint for receiving queries.
    spa_params : dict
        Configuration passed to all SinglePurposeAgent instances.
    loop : asyncio.AbstractEventLoop
        Main event loop for supervisor execution.
    max_steps : int
        Maximum LangGraph steps for supervisor and agents.

    Notes
    -----
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
        7. Starts agent executor consumer thread (event-driven).

        After successful initialization, the node is ready to receive user queries
        via ROS2 service and manage the hierarchical multi-agent execution.

        Notes
        -----
        Various exceptions from parent class or SupervisorManager if
        configuration loading, LLM connection, or LangGraph building fails.

        """
        # Call the base class initializer
        super().__init__()

        self.get_spa_params()

        # Create the service to listen for user queries
        self._srv_group = MutuallyExclusiveCallbackGroup()
        self.agent_srv = self.create_service(
            srv_type=CallAgent,
            srv_name=self.service_name,
            callback=self.agent_callback,
            callback_group=self._srv_group
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
            spa_params=self.spa_params,
            agent_timeout=self.agent_timeout,
            max_finished_history=self.max_finished_history,
            max_agents_in_context=self.max_agents_in_context
        )

        # Retrieve tools for Ollama agent
        self.loop.run_until_complete(
            self.supervisor_manager.ollama_agent.retrieve_tools(
                self.supervisor_manager.supervisor_tools
            ))

        # Build the LangGraph workflow
        self.build_graph()

        # Start the event-driven agent consumer thread
        self.supervisor_manager.executor.start()

        self.get_logger().info('Hierarchical Multiagent LangGraph Node has been started.')

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

        Uses supervisor_manager and the main event loop rather than explicit
        parameters.

        Returns
        -------
        None
            Compiled graph stored in supervisor_manager.graph

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

    def _process_graph(self, input_state: InputState, thread_id: str) -> None:
        """
        Invoke supervisor's graph synchronously, blocking the calling thread.

        Submits the supervisor task to the event loop and blocks until completion.
        Results are logged to console after processing finishes.

        Parameters
        ----------
        input_state : InputState
            Initial state with 'user_prompt'.
            Format: {'user_prompt': 'user query string'}
        thread_id : str
            Checkpoint thread identifier for state persistence.

        Returns
        -------
        None

        Side Effects:
            - Blocks calling thread until graph execution completes
            - Logs result/errors to console (not returned to ROS2 caller)
            - May trigger agent creation/execution in background
            - Updates supervisor_manager agent lists during execution

        """
        config = {'configurable': {'thread_id': thread_id}}
        self.loop.run_until_complete(
            self.supervisor_manager.graph.ainvoke(input_state, config=config)
        )

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
            5. Call _process_graph(input_state, thread_id)
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

        Parameters
        ----------
        request : CallAgent.Request
            Service request with query string.
        response : CallAgent.Response
            Service response object to populate.

        Returns
        -------
        CallAgent.Response
            Response with fixed agent_response string.

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

        self.get_logger().info(f'Received user query: [{user_query}]')

        # Use a fixed thread ID for supervisor to maintain context
        thread_id = 'supervisor'

        # Create input state with user prompt
        input_state: InputState = {'user_prompt': user_query}

        # Process graph
        self.get_logger().info('Processing query asynchronously...')
        self._process_graph(input_state, thread_id)
        response.agent_response = 'Query submitted for processing'

        self.get_logger().info('Query processing submitted; returning response to caller.')
        return response

    def get_spa_params(self) -> None:
        """
        Declare and retrieve ROS2 parameters for SinglePurposeAgent configuration.

        Iterates over ``_SPA_PARAM_DEFS`` to declare each parameter with its
        default value, retrieve the actual value from the ROS2 parameter server,
        and log it. The correct ``ParameterValue`` accessor is inferred from the
        Python type of the default value via ``_VALUE_ACCESSORS``.

        Additionally declares the ``agent_timeout`` parameter (float) which
        controls the maximum execution time for a single agent.
        """
        # Initialize spa_params dictionary by copying agent_params
        self.spa_params = self.agent_params.copy()
        # Remove mcp_client from spa_params if it exists
        if 'mcp_client' in self.spa_params:
            del self.spa_params['mcp_client']

        # Declare and retrieve all SPA parameters from the data-driven table
        for ros_name, dict_key, default in _SPA_PARAM_DEFS:
            self.declare_parameter(ros_name, default)
            accessor = _VALUE_ACCESSORS[type(default)]
            value = getattr(
                self.get_parameter(ros_name).get_parameter_value(), accessor)
            self.spa_params[dict_key] = value
            self.get_logger().info(
                f'The parameter {ros_name} is set to: [{value}]')

        # Declare and retrieve agent execution timeout (seconds)
        self.declare_parameter('agent_timeout', 120.0)
        self.agent_timeout = self.get_parameter(
            'agent_timeout').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter agent_timeout is set to: [{self.agent_timeout}]')

        # Declare and retrieve the finished-agents history retention limits
        self.declare_parameter('max_finished_history', 20)
        self.max_finished_history = self.get_parameter(
            'max_finished_history').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_finished_history is set to: '
            f'[{self.max_finished_history}]')

        self.declare_parameter('max_agents_in_context', 5)
        self.max_agents_in_context = self.get_parameter(
            'max_agents_in_context').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_agents_in_context is set to: '
            f'[{self.max_agents_in_context}]')


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
        - Agent consumer thread runs in background, triggered by events
        - Agent callback (agent_callback) handles service requests from multiple callers
        - Multiple agents can execute in parallel via dedicated threads

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
           - Starts event-driven agent consumer thread
        2. Executes until shutdown
        3. Cleanup: stops consumer thread, destroys node, shuts down rclpy

    Parameters
    ----------
    args : list | None
        Command-line arguments passed to rclpy.init().
        None uses sys.argv. Typical: ['--ros-args', '--log-level', 'info']

    Returns
    -------
    None
        Exits process via rclpy.shutdown() or exception

    Side Effects:
        - Initializes global ROS2 context (rclpy.init)
        - Creates HierarchicalMultiagent node
        - Registers ROS2 service and starts consumer thread
        - Blocks until shutdown (run forever in normal operation)
        - Prints shutdown reason to console

    Usage:
        $ ros2 run hierarchical_multiagent_langgraph supervisor
        # Node runs until Ctrl+C or external shutdown signal

    """
    rclpy.init(args=args)

    agent = None
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
    finally:
        if agent is not None:
            # Stop the event-driven consumer thread before destroying the node
            agent.supervisor_manager.executor.stop()
            agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
