from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Messages, Ollama
from langsmith import traceable


class MyState(Messages):
    sdsdsds


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
            ollama_agent: Ollama = None,
            max_steps: int = 5):
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

    @traceable
    async def pass_query_to_agent(self, state: MyState) -> MyState:
        # Implement the logic to pass the query to the agent

        # Invoke Ollama agent
        try:
            state = await self.ollama_agent.invoke(state)
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
        # Define your custom workflow here
        workflow = StateGraph(Messages)
        # Add nodes
        workflow.add_node('pass_query_to_agent', self.pass_query_to_agent)
        workflow.add_node('finalize_conversation', self.finalize_conversation)

        # Add edges between nodes
        workflow.add_edge(START, 'pass_query_to_agent')
        # After an agent step, check end conditions and proceed accordingly
        workflow.add_conditional_edges(
            'pass_query_to_agent',
            self.route_on_tool_call,
            {
                'agent': 'pass_query_to_agent',
                'finish': 'finalize_conversation'
            },
        )
        workflow.add_edge('finalize_conversation', END)
        # Compile the workflow into an executable graph
        self.graph = workflow.compile()
