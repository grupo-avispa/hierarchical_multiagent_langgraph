

import asyncio
import time

from hierarchical_multiagent_langgraph.supervisor import InputState, SupervisorManager
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from llm_interactions_msgs.srv import CallAgent

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


class HierarchicalMultiagent(LangGraphRosBase):

    def __init__(self):
        """Initialize the LangGraph ROS node."""
        # Call the base class initializer
        super().__init__('hierarchical_multiagent_node')

        # Initialize the Supervisor Manager
        self.supervisor_manager = SupervisorManager(
            system_prompt=self.system_prompt,
            logger=self.get_logger(),
            ollama_agent=self.ollama_agent,
            max_steps=self.max_steps)

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
        asyncio.create_task(
            self.supervisor_manager.graph.ainvoke(input_state, config=config)
        )

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
