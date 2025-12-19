from enum import Enum
from pathlib import Path
from langgraph.graph import START, StateGraph, END
from langchain.tools import tool
from langsmith import traceable
from ollama import Message
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama
from langgraph_base_ros.chat_template_render import Messages

class AgentStatus(str, Enum):
    """Enumeration of possible agent statuses."""

    IDLE = 'idle'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'

class SinglePurposeAgent(LangGraphBase):
    """
    Represents a single-purpose agent with its configuration and status.

    Attributes:
        agent_id: Unique identifier for the agent.
        query: The task or query assigned to this agent.
        status: Current status of the agent (IDLE, RUNNING, SUCCESS, or FAILURE).
    """

    def __init__(
            self,
            logger=None,
            ollama_agent: Ollama | None = None,
            max_steps: int = 5
    ) -> None:
        """
        Initialize the Agent.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).
            ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
            max_steps (int): Maximum allowed steps before finishing interaction.

        Returns:
            None
        """
        if ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')
        
        super().__init__(
            logger=logger,
            ollama_agent=ollama_agent,
            max_steps=max_steps)

        self.id: int = -1  # Unique identifier for the agent
        self.status: AgentStatus = AgentStatus.IDLE  # Current status of the agent
        self._generate_tools_list() # Generate tools list for the agent
        self._get_system_prompt() # Load system prompt
        # Note: retrieve_tools must be called asynchronously after initialization

    def set_id(self, agent_id: int) -> None:
        """
        Set the unique identifier for the agent.

        Parameters:
            agent_id (int): Unique identifier to assign to the agent.

        Returns:
            None
        """
        self.id = agent_id

    def get_id(self) -> int:
        """
        Get the unique identifier of the agent.

        Returns:
            int: The unique identifier of the agent.
        """
        return self.id

    def get_status(self) -> str:
        """
        Get the current status of the agent.

        Returns:
            str: The current status of the agent.
        """
        return self.status

    def set_status(self, status: AgentStatus) -> None:
        """
        Set the current status of the agent.

        Parameters:
            status (AgentStatus): The new status to assign to the agent.

        Returns:
            None
        """
        self.status = status

    def _get_system_prompt(self) -> str:
        """
        Retrieve the system prompt for the agent.
        Returns:
            None: Sets the system prompt attribute.
        """
        # Get the templates directory path relative to this file
        current_dir = Path(__file__).parent
        # templates_path = str(current_dir.parent / 'templates')
        templates_path = "/home/oscar/colcon_ws/src/interaction/hierarchical_multiagent_langgraph/templates"
        try:
            with open(templates_path + '/agent_system_prompt.jinja', 'r') as f:
                self.sys_prompt = f.read()
        except FileNotFoundError:
            self._log(f"Agent system prompt template not found at path: {templates_path + '/agent_system_prompt.jinja'}")
            self.sys_prompt = "You are a helpful assistant designed to perform specific tasks."
        return self.sys_prompt

    # ========== LANGGRAPH NODES ==========

    @traceable
    async def query_response(self, state: Messages) -> Messages:
        """
        Generate LLM response based on conversation state.
        Receives the current conversation message list from ollama agent
        and updates state with LLM response.

        Parameters:
            state (Messages): Current conversation state with messages.

        Returns:
            Messages: Updated state with agent response.
        """
        self.status = AgentStatus.RUNNING 
        # print( f"AGENT {self.id}: Invoking Ollama agent with state: {state}" )
        # Invoke Ollama agent
        try:
            # Check if any of the message roles is 'system'
            has_sys_message = any(
                (msg['role'] == 'system' and msg['content'] is not None)
                for msg in state['messages'])
            if not has_sys_message:
                # Prepend system prompt if not already present
                state['messages'].insert(0, Message(
                    role='system',
                    content=self.sys_prompt
                ))
            self.state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log(f"AGENT: Error during Ollama agent invocation: {e}")
            raise e
        
        return self.state
    
    @traceable
    def manage_steps(self, state: Messages) -> str:
        """
        Determine the next step in the conversation flow.

        Checks if the last message contains a tool call to decide whether
        to continue querying or finish the interaction.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            str: Next node to transition to ('query_response' or 'finish_ollama_interaction').
        """
        self.steps += 1
        uc = 'finish'
        self._log(f"AGENT: Managing steps, current step: {self.steps}")
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                if self.steps < self.max_steps:
                    uc = 'agent'
                else:
                    self._log("AGENT: Maximum steps reached, finishing interaction.")
            else:
                self._log("AGENT: No tool call detected, finishing interaction.")
                self._log("AGENT: Final response from assistant:\n" +
                        f"{state['messages'][-1]['content']}")
            # Update messages count
            self.messages_count = len(state['messages'])
            self._log(f"AGENT: Total messages in conversation: {self.messages_count}")
        except Exception as e:
            self._log(f"AGENT: Error in manage_steps: {e}")
            uc = 'finish'
        return uc
    
    @traceable
    async def finish_ollama_interaction(self, state: Messages) -> Messages:
        """
        Finalize the Ollama interaction and return the final response.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            Messages: Final state after finishing interaction.
        """

        self._log("Finalizing Ollama interaction.")
        if self.steps >= self.max_steps:
            self._log("AGENT: Maximum steps reached during finalization.")
            self.status = AgentStatus.FAILURE
        else:
            self._log("AGENT: Agent reached final state before maximum steps.")
            self.status = AgentStatus.SUCCESS
        self.steps = 0
        self.ollama_agent.reset_memory()
        return state
    
    # ========== GRAPH GENERATION ==========

    async def make_graph(self):
        """
        Initialize and compile the LangGraph workflow.

        This method creates a LangGraph StateGraph with nodes for query processing and
        conversation finalization. It defines the flow of the conversation based on
        LLM outputs and compiles the graph for execution.

        Returns:
            None: The compiled graph is stored in self.graph.
        """

        # Create the StateGraph workflow
        workflow = StateGraph(Messages)

        # Add graph nodes:
        # - query_response: Main LLM reasoning node
        workflow.add_node('query_response', self.query_response)
        # - finish_ollama_interaction: Final node to end interaction
        workflow.add_node('finish_ollama_interaction', self.finish_ollama_interaction)

        # Define graph edges and flow:
        # After start, proceed to query response
        workflow.add_edge(START, 'query_response')
        # After a agent step, check end conditions and proceed accordingly
        workflow.add_conditional_edges(
            'query_response',
            self.manage_steps, 
            {'agent': 'query_response', 'finish': 'finish_ollama_interaction'},
        )

        # Compile the graph workflow
        self.graph = workflow.compile()
    
    # ========== LANGGRAPH TOOLS ==========
    
    @staticmethod
    @tool("find_object",
          description="Find the location of a specified object.",
          args_schema={"type": "object",
                       "properties": {
                           "object_name": {
                               "type": "string",
                               "description": "Name of the object to find."
                           }
                       },
                       "required": ["object_name"]})
    def find_object(object_name: str) -> str:
        """
        Find the location of a specified object.

        Parameters:
            object_name (str): Name of the object to find.
        Returns:
            str: Location of the object.
        """
        # Implement the logic to find the object here
        location = f"Object {object_name} is located in the kitchen."
        return location
    
    @staticmethod
    @tool("retrieve_information",
          description="Retrieve information on a given topic.",
          args_schema={"type": "object",
                       "properties": {
                           "topic": {
                               "type": "string",
                               "description": "Topic to retrieve information about."
                           }
                       },
                       "required": ["topic"]})
    def retrieve_information(topic: str) -> str:
        """
        Retrieve information on a given topic.

        Parameters:
            topic (str): Topic to retrieve information about.
        Returns:
            str: Retrieved information.
        """
        # Implement the logic to retrieve information here
        info = f"Information about {topic}: It is a fascinating subject!"
        return info