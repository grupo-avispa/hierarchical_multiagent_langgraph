

import time

from hierarchical_multiagent_langgraph.supervisor import SupervisorManager
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from langgraph_base_ros.ollama_utils import Messages
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

    def agent_callback(self, request, response):
        """
        Handle incoming service requests.

        Receives a user request, processes it using the agent graph,
        and returns the generated response.

        Parameters:
            request: The CallAgent request containing query details.
            response: The CallAgent response to be populated with the agent's reply.
        Returns:
            The populated response with the agent's generated reply.
        """
        user_query = request.query
        response_needed = request.response_needed

        self.get_logger().debug(f'Received user query:\n{user_query}')

        init_time = time.time()

        # Prepare the initial conversation state with system prompt and user query
        initial_state: Messages = {
            'messages': [
                self.ollama_agent.create_message(
                    role='system',
                    content=self.system_prompt
                )
            ]
        }
        initial_state['messages'].append(
            self.ollama_agent.create_message(
                role='user',
                content=user_query
            )
        )
        # Run the agent graph asynchronously
        result = self.loop.run_until_complete(
            self.graph_manager.graph.ainvoke(initial_state)
        )

        # Log processing time and generated response
        self.get_logger().info(f'Agent processing time: {time.time() - init_time:.3f} seconds')
        self.get_logger().info(f'Generated response: {result["messages"][-1].get("content", "")}')

        # Return the response to the user request
        response.response_text = result['messages'][-1].get('content', '')
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
