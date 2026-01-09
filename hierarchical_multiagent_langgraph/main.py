

import asyncio
import time

from hierarchical_multiagent_langgraph.supervisor import InputState, SupervisorManager
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from llm_interactions_msgs.srv import CallAgent
from langgraph_base_ros.ollama_utils import Ollama

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


class HierarchicalMultiagent(LangGraphRosBase):

    def __init__(self):
        """Initialize the LangGraph ROS node."""
        # Call the base class initializer
        super().__init__()

        self.get_spa_params()

        try:
            self.loop.run_until_complete(self.initialize_mcp_client(
                self.spa_mcp_servers,
                self.spa_params
            ))
        except Exception as e:
            self.get_logger().error(f'Failed to initialize mcp client for SPA: {e}')
            raise


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

        # Create the subscriber to listen for user queries
        self.group = ReentrantCallbackGroup()
        self.agent_srv = self.create_service(
            srv_type=CallAgent,
            srv_name=self.service_name,
            callback=self.agent_callback,
            callback_group=self.group
        )

        self.get_logger().info('Hierarchical Multiagent LangGraph Node has been started.')

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
        if "mcp_client" in self.spa_params:
            del self.spa_params["mcp_client"]

        # Declare and retrieve MCP servers parameter
        self.declare_parameter('spa_mcp_servers', 'mcp.json')
        self.spa_mcp_servers = self.get_parameter(
            'spa_mcp_servers').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_mcp_servers is set to: [{self.spa_mcp_servers}]')
        
        # Declare and retrieve system prompt template path parameter
        self.declare_parameter('spa_system_prompt_file', 'system_prompt.jinja')
        self.spa_params["system_prompt_file"] = self.get_parameter(
            'spa_system_prompt_file').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_system_prompt_file is set to: [{self.spa_params["system_prompt_file"]}]')

        # Declare and retrieve model chat template file path parameter
        self.declare_parameter('spa_template_type', 'qwen3')
        self.spa_params["template_type"] = self.get_parameter(
            'spa_template_type').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_template_type is set to: [{self.spa_params["template_type"]}]')

        # Declare and retrieve LLM model name parameter
        self.declare_parameter('spa_llm_model', 'qwen3:0.6b')
        self.spa_params["model"] = self.get_parameter(
            'spa_llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_llm_model is set to: [{self.spa_params["model"]}]')
        # Declare tool call regex pattern to extract tool calls from LLM response
        self.declare_parameter('spa_tool_call_pattern', '<tool_call>(.*?)</tool_call>')
        self.spa_params["tool_call_pattern"] = self.get_parameter(
            'spa_tool_call_pattern').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter spa_tool_call_pattern is set to: [{self.spa_params["tool_call_pattern"]}]')

        # Declare and retrieve LangGraph workflow parameters
        self.declare_parameter('spa_max_steps', 5)
        self.spa_params["max_steps"] = self.get_parameter(
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
