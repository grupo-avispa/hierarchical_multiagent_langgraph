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

import asyncio
import threading
import time

from hierarchical_multiagent_langgraph.supervisor import (
    InputState,
    RunningAgentsState,
    SupervisorManager
)
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from llm_interactions_msgs.srv import CallAgent


import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


class HierarchicalMultiagent(LangGraphRosBase):
    """ROS2 node for hierarchical multi-agent LangGraph coordination.

    This node implements a supervisor-based multi-agent system where a supervisor
    agent decomposes complex user queries into subtasks and coordinates multiple
    single-purpose agents (SPAs) to execute them. Each agent runs in its own
    thread with an independent event loop to ensure concurrency and isolation.

    The node exposes a ROS2 service for receiving user queries and manages:
    - Agent instantiation and lifecycle
    - Task delegation to appropriate agents
    - Response synthesis from multiple agent outputs
    - Asynchronous and blocking execution modes

    Attributes:
        supervisor_manager (SupervisorManager): Manages the supervisor agent
            and the LangGraph workflow for task orchestration.
        agent_lists_lock (threading.Lock): Synchronizes access to agent lists
            between the main thread and agent execution threads.
        pending_agents_list (list): Queue of agents waiting to be executed.
        running_agents_list (list): List of currently executing agents.
        agent_srv (rclpy.node.Service): ROS2 service for receiving user queries.
        agent_timer (rclpy.node.TimerHandle): Timer for consuming pending agents.
        spa_params (dict): Configuration parameters for single-purpose agents.
    """

    def __init__(self):
        """Initialize the Hierarchical Multiagent ROS2 node.

        Sets up the supervisor manager, builds the LangGraph workflow,
        creates ROS2 service for handling queries, and initializes the
        agent execution timer.
        """
        # Call the base class initializer
        super().__init__()

        self.get_spa_params()

        # Initialize the Supervisor Manager
        self.supervisor_manager = SupervisorManager(
            logger=self.get_logger(),
            ollama_agent=self.ollama_agent,
            max_steps=self.max_steps,
            system_prompt_path=self.system_prompt_file,
            spa_params=self.spa_params,
            loop=self.loop
        )

        # Retrieve tools for Ollama agent
        self.loop.run_until_complete(
            self.supervisor_manager.ollama_agent.retrieve_tools(
                self.supervisor_manager.supervisor_tools
            ))

        # Build the LangGraph workflow
        self.build_graph()

        # Create the subscriber to listen for user queries
        self.group = ReentrantCallbackGroup()
        self.agent_srv = self.create_service(
            srv_type=CallAgent,
            srv_name=self.service_name,
            callback=self.agent_callback,
            callback_group=self.group
        )

        # Create timer to consume pending agents from the queue
        # Uses ReentrantCallbackGroup to allow concurrent execution
        self.agent_timer = self.create_timer(
            1.0,  # Timer period in seconds
            self._agent_execution_timer_callback,
            callback_group=self.group
        )

        self.get_logger().info('Hierarchical Multiagent LangGraph Node has been started.')

    def _agent_execution_timer_callback(self) -> None:
        """
        Timer callback to consume and execute pending agents.

        This callback is triggered periodically by the ROS2 timer. It checks
        the pending agents queue and spawns a new thread to execute each
        pending agent with its own event loop. This ensures agents run
        independently from the supervisor's event loop.

        Returns:
            None
        """
        # Try to get a pending agent from the list
        agent_idle = None
        self.supervisor_manager.agent_lists_lock.acquire()
        if len(self.supervisor_manager.pending_agents_list) > 0:
            agent_idle = self.supervisor_manager.pending_agents_list.pop(0)
        self.supervisor_manager.agent_lists_lock.release()

        if agent_idle is not None:
            agent_id = agent_idle.agent.get_id()
            self.get_logger().info(
                f'Timer: Starting execution of agent {agent_id} in thread '
                f'[{threading.current_thread().name}]')
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Schedule the agent execution coroutine in the new event loop
            agent_task = loop.create_task(
                self.supervisor_manager.run_agent(
                    agent_idle.agent,
                    agent_idle.input_state
                ))

            # Create running agent object with all required fields
            running_agent = RunningAgentsState(
                agent_id=agent_id,
                input_prompt=agent_idle.input_state['messages'][0]['content'],
                coroutine_handler=agent_task,
                event_loop=loop
            )

            # Add to running agents list
            self.supervisor_manager.agent_lists_lock.acquire()
            self.supervisor_manager.running_agents_list.append(running_agent)
            self.supervisor_manager.agent_lists_lock.release()

            # Need to await the agent task to completion
            try:
                self.get_logger().info(
                    f'Working on agent [{agent_id}]'
                    f' in thread [{threading.current_thread().name}]...')
                loop.run_until_complete(agent_task)
            except asyncio.CancelledError:
                self.get_logger().info(f'Agent {agent_id} execution was cancelled.')

    def build_graph(self) -> None:
        """
        Initialize and compile the LangGraph workflow.

        Returns:
            None
        """
        # Initialize and compile the LangGraph workflow
        try:
            self.loop.run_until_complete(self.supervisor_manager.make_graph())
        except Exception as e:
            self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
            raise

        self.get_logger().info('SupervisorManager graph created successfully...')

    def _process_graph(self, input_state: InputState, thread_id: str) -> dict:
        """
        Process the agent graph with the given input state.

        Parameters:
            input_state: The input state containing user prompt.
            thread_id: The thread ID for checkpoint persistence.

        Returns:
            dict: The result from the graph invocation.
        """
        config = {'configurable': {'thread_id': thread_id}}
        return self.loop.run_until_complete(
            self.supervisor_manager.graph.ainvoke(input_state, config=config)
        )

    def _process_graph_async(self, input_state: InputState, thread_id: str) -> None:
        """
        Process the agent graph asynchronously without blocking.

        Parameters:
            input_state: The input state containing user prompt.
            thread_id: The thread ID for checkpoint persistence.

        Returns:
            None
        """
        config = {'configurable': {'thread_id': thread_id}}
        result = self.loop.run_until_complete(
            self.supervisor_manager.graph.ainvoke(input_state, config=config)
        )
        print(f'Asynchronous processing result: {result}')

    def agent_callback(self, request, response):
        """
        Handle incoming service requests.

        Receives a user request, processes it using the agent graph,
        and returns the generated response. The processing behavior depends
        on the response_needed flag: if True, blocks until completion and
        returns the response; if False, processes asynchronously.

        Parameters:
            request: The CallAgent request containing query details.
            response: The CallAgent response to be populated with the agent's reply.
        Returns:
            The populated response with the agent's generated reply.
        """
        user_query = request.query
        response_needed = request.response_needed

        self.get_logger().debug(f'Received user query:\n{user_query}')

        # Use a fixed thread ID for supervisor to maintain context
        thread_id = 'supervisor'

        init_time = time.time()

        # Create input state with user prompt
        input_state: InputState = {'user_prompt': user_query}

        # Process graph based on response_needed flag
        if response_needed:
            # Blocking behavior: wait for the result and return it
            result = self._process_graph(input_state, thread_id)

            # Log processing time and generated response
            self.get_logger().info(
                f'Agent processing time: {time.time() - init_time:.3f} seconds'
            )
            self.get_logger().info(
                f'Generated response: {result["messages"][-1].get("content", "")}'
            )

            # Return the response to the user request
            response.agent_response = result['messages'][-1].get('content', '')
        else:
            # Non-blocking behavior: schedule asynchronously
            self.get_logger().info('Processing query asynchronously (response not needed)')
            self._process_graph_async(input_state, thread_id)
            response.agent_response = 'Query submitted for processing'

        return response

    def get_spa_params(self) -> None:
        """
        Retrieve and configure ROS2 parameters relative to single purpose agents creation.

        Declares and retrieves parameters from the ROS2 parameter server,
        Logs each parameter value for verification.

        Parameters:
            None

        Returns:
            None
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

        # Declare and retrieve LangGraph workflow parameters
        self.declare_parameter('spa_max_steps', 5)
        self.spa_params['max_steps'] = self.get_parameter(
            'spa_max_steps').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter spa_max_steps is set to: [{self.spa_params["max_steps"]}]')


def main(args=None) -> None:
    """
    Run the ROS2 agent.

    Initialize the ROS2 context, create the agent node, and spin
    until shutdown is requested. Uses a MultiThreadedExecutor
    for concurrent callbacks.

    Parameters:
        args: Command-line arguments (optional).

    Returns:
        None
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
