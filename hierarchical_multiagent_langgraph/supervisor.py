from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Messages, Ollama
from langsmith import traceable
from typing import TypedDict
from enum import Enum


class AgentStatus(str, Enum):
    """Enumeration of possible agent statuses."""

    IDLE = 'idle'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class SinglePurposeAgent(TypedDict):
    """
    Represents a single-purpose agent with its configuration and status.

    Attributes:
        agent_id: Unique identifier for the agent.
        query: The task or query assigned to this agent.
        status: Current status of the agent (IDLE, RUNNING, SUCCESS, or FAILURE).
    """

    agent_id: int
    query: str
    status: AgentStatus


class ContextState(Messages):
    """
    Represents the context state of a conversation with hierarchical agents.

    This state extends the base Messages state to include additional context
    information about the date and the list of single-purpose agents being managed.

    Attributes:
        date: The date or timestamp associated with the conversation context.
        agents: List of single-purpose agents being managed in this context.
    """

    date: str
    agents: list[SinglePurposeAgent]


@tool
def create_agent(agent_id: int, query: str) -> str:
    """Create an agent based on the provided query."""
    return f'Agent {agent_id} created with query: {query}'


@tool
def delete_agent(agent_id: int) -> str:
    """Delete an agent by its ID."""
    return f'Agent {agent_id} deleted successfully.'


@tool
def skip_agent() -> str:
    """Skip the current agent action."""
    return 'Agent action skipped.'


class SupervisorManager(LangGraphBase):
    def __init__(
            self,
            logger=None,
            ollama_agent: Ollama | None = None,
            max_steps: int = 5
    ) -> None:
        """
        Initialize the Supervisor Manager.

        Creates the LLM instance with default configuration.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).

        Returns:
            None
        """
        super().__init__(
            logger=logger,
            ollama_agent=ollama_agent,
            max_steps=max_steps)
        if self.ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')
        self.ollama_agent: Ollama = self.ollama_agent
        self.initial_state: Messages = {'messages': []}

    def set_initial_state(self, system_prompt: str, user_query: str = '') -> Messages:
        """
        Set and return the initial conversation state with a system prompt and optional user query.

        Parameters:
            system_prompt: The system prompt to initialize the conversation.
            user_query: The user query to append to the initial state (optional).

        Returns:
            Messages: The initialized conversation state.
        """
        self.initial_state = {
            'messages': [
                self.ollama_agent.create_message(
                    role='system',
                    content=system_prompt
                )
            ]
        }
        if user_query:
            self.initial_state['messages'].append(
                self.ollama_agent.create_message(
                    role='user',
                    content=user_query
                )
            )
        return self.initial_state

    @traceable
    async def process_agent_query(self, state: ContextState) -> ContextState:
        """
        Process the current query through the Ollama agent.

        Invokes the Ollama agent to generate a response based on the current
        conversation state. This method handles agent invocation and error
        management during query processing.

        Parameters:
            state: The current context state containing messages and agent information.

        Returns:
            ContextState: The updated state after agent processing.

        Raises:
            ValueError: If the Ollama agent invocation fails.
        """
        try:
            state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log(f'Error during Ollama agent invocation: {e}')
            raise e

        return state

    @traceable
    def route_on_tool_call(self, state: Messages) -> str:
        """
        Route the conversation flow based on tool call presence.

        Checks if the last message contains a tool call to determine whether
        to continue the agent loop or finish the interaction. Also enforces
        the maximum step limit to prevent infinite loops.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            str: Next node to transition to ('agent' to continue, 'finish' to end).
        """
        self.steps += 1
        uc = 'finish'
        self._log(f'Managing steps, current step: {self.steps}')
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                if self.steps < self.max_steps:
                    uc = 'agent'
                else:
                    self._log('Maximum steps reached, finishing interaction.')
            else:
                self._log('No tool call detected, finishing interaction.')
                self._log(
                    'Final response from assistant:\n' + f"{state['messages'][-1]['content']}")
            # Update messages count
            self.messages_count = len(state['messages'])
            self._log(f'Total messages in conversation: {self.messages_count}')
        except Exception as e:
            self._log(f'Error in manage_steps: {e}')
            uc = 'finish'
        return uc

    @traceable
    async def finalize_conversation(self, state: Messages) -> Messages:
        """
        Finalize the conversation and clean up resources.

        Resets the step counter and clears the agent's memory after the
        conversation completes, either due to reaching the final response
        or hitting the maximum step limit.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            Messages: Final state after cleanup and finalization.
        """
        self._log('Finalizing Ollama interaction.')
        if self.steps >= self.max_steps:
            self._log('Maximum steps reached during finalization.')
        else:
            self._log('Agent reached final state before maximum steps.')
        self.steps = 0
        self.ollama_agent.reset_memory()
        return state

    async def make_graph(self):
        """
        Build and compile the LangGraph workflow.

        Constructs a state graph with nodes for processing queries and finalizing
        conversations, along with conditional edges to route the flow based on
        tool call detection.

        Returns:
            None
        """
        # Define your custom workflow here
        workflow = StateGraph(Messages)
        # Add nodes
        workflow.add_node('process_agent_query', self.process_agent_query)
        workflow.add_node('finalize_conversation', self.finalize_conversation)

        # Add edges between nodes
        workflow.add_edge(START, 'process_agent_query')
        # After an agent step, check end conditions and proceed accordingly
        workflow.add_conditional_edges(
            'process_agent_query',
            self.route_on_tool_call,
            {
                'agent': 'process_agent_query',
                'finish': 'finalize_conversation'
            },
        )
        workflow.add_edge('finalize_conversation', END)
        # Compile the workflow into an executable graph
        self.graph = workflow.compile()
