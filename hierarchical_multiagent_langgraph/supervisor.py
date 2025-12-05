import asyncio
from typing import TypedDict

from jinja2 import Template
from hierarchical_multiagent_langgraph.agent import SinglePurposeAgent, AgentStatus
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Messages, Ollama
from langsmith import traceable


class InputState(TypedDict):
    """
    Represents the input state for the supervisor manager.

    Attributes:
        user_prompt: The user prompt for the conversation.
    """

    user_prompt: str


class ContextState(Messages):
    """
    Represents the context state of a conversation with hierarchical agents.

    This state extends the base Messages state to include metadata
    about active agents being managed.

    Attributes:
        agents: Dictionary of active agents indexed by agent_id.
        next_agent_id: Counter for assigning unique agent IDs.
    """

    agents: dict[int, dict]
    next_agent_id: int


class SupervisorManager(LangGraphBase):

    def __init__(
        self,
        system_prompt: str,
        logger=None,
        ollama_agent: Ollama | None = None,
        max_steps: int = 5
    ) -> None:
        """
        Initialize the Supervisor Manager.

        Creates the LLM instance with default configuration.

        Parameters:
            system_prompt: The system prompt for the supervisor.
            logger: Optional ROS2 logger to use for logging (default: None).
            ollama_agent: Optional Ollama agent instance (default: None).
            max_steps: Maximum number of steps for the graph execution (default: 5).

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
        # Dictionary to store active agent instances
        self.active_agents: dict[int, SinglePurposeAgent] = {}
        # Store prompt
        self.system_prompt: str = system_prompt

    @tool(
        'create_agent',
        description='Creates a new agent to handle a specific task.')
    def create_agent(self, state: ContextState, query: str) -> str:
        """
        Create a new agent to handle the specified task.

        Parameters:
            state: The current context state.
            query: The task description for the new agent.

        Returns:
            str: Confirmation message with the assigned agent ID.
        """
        agent_id = state['next_agent_id']

        self._log(f'Creating agent {agent_id} for task: {query}')

        # Create agent instance
        new_agent = SinglePurposeAgent(
            logger=self.logger,
            ollama_agent=self.ollama_agent,
            max_steps=self.max_steps
        )
        new_agent.set_id(agent_id)
        new_agent.query = query
        new_agent.set_status(AgentStatus.RUNNING)

        # Store agent info in state
        state['agents'][agent_id] = {
            'id': agent_id,
            'query': query,
            'status': AgentStatus.RUNNING
        }

        # Store agent instance
        self.active_agents[agent_id] = new_agent

        # Increment agent counter
        state['next_agent_id'] += 1

        # Invoke agent graph asynchronously
        asyncio.create_task(self._run_agent(new_agent))

        self._log(f'Agent {agent_id} created and started asynchronously')

        return f'Agent {agent_id} created successfully for task: {query}'

    @tool(
        'delete_agent',
        description='Deletes an existing agent by its ID.')
    def delete_agent(self, state: ContextState, agent_id: int) -> str:
        """
        Delete an existing agent by its ID.

        Parameters:
            state: The current context state.
            agent_id: The ID of the agent to delete.

        Returns:
            str: Confirmation message.
        """
        if agent_id in state['agents']:
            self._log(f'Deleting agent {agent_id}')
            agent_info = state['agents'][agent_id]
            del state['agents'][agent_id]
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            return f'Agent {agent_id} deleted successfully (was working on: {agent_info["query"]})'
        else:
            self._log(f'Agent {agent_id} not found')
            return f'Error: Agent {agent_id} not found'

    @tool(
        'skip_agent',
        description='Skip agent management for this iteration.')
    def skip_agent(self) -> str:
        """
        Skip agent management for this iteration.

        Returns:
            str: Confirmation message.
        """
        self._log('Skipping agent action')
        return 'No agent action needed for this request'

    async def _run_agent(self, agent: SinglePurposeAgent) -> None:
        """
        Run an agent's graph asynchronously.

        Parameters:
            agent: The agent instance to run.

        Returns:
            None
        """
        try:
            # Build the agent's graph if not already built
            if agent.graph is None:
                await agent.make_graph()

            # Run the agent's graph
            # This is a placeholder - implement actual graph invocation
            self._log(f'Agent {agent.get_id()} executing task: {agent.query}')

            # Update agent status
            agent.set_status(AgentStatus.SUCCESS)

        except Exception as e:
            self._log(f'Error running agent {agent.get_id()}: {e}')
            agent.set_status(AgentStatus.FAILURE)

    @traceable
    async def set_initial_messages(self, state: InputState) -> ContextState:
        """
        Set initial messages from stored prompts and return the initial conversation state.

        This node processes the input state and creates the initial context state
        with system and user prompts as the first messages.

        Parameters:
            state: The input state containing optional user prompt.

        Returns:
            ContextState: The initialized conversation state with system and user messages.
        """
        # Create initial context state
        context_state: ContextState = {
            'messages': [
                self.ollama_agent.create_message(
                    role='system',
                    content=self.system_prompt
                )
            ],
            'agents': {},
            'next_agent_id': 1
        }

        # Add user message if user prompt is set
        if state.get('user_prompt'):
            context_state['messages'].append(
                self.ollama_agent.create_message(
                    role='user',
                    content=state['user_prompt']
                )
            )

        return context_state

    @traceable
    async def analyze_task(self, state: ContextState) -> ContextState:
        """
        Analyze the incoming task and invoke the LLM to make supervisor decisions.

        This node:
        1. Obtains context about current active agents.
        2. Retrieves the system message from the initial messages.
        3. Renders the system prompt with current agent context.
        4. Calls the LLM with the updated messages.

        Parameters:
            state: The current context state.

        Returns:
            ContextState: Updated state with LLM response.
        """
        self._log('Analyzing task and processing supervisor decision...')

        # Build context about current agents
        agents_list = []
        if state['agents']:
            for agent_id, agent_info in state['agents'].items():
                agents_list.append({
                    'id': agent_id,
                    'query': agent_info['query'],
                    'status': agent_info['status']
                })

        # Find and extract the system message (first message with role='system')
        system_message_content = None
        for msg in state['messages']:
            if msg.get('role') == 'system':
                system_message_content = msg.get('content')
                break

        # Render the system prompt with agent context using Jinja2
        template = Template(system_message_content or '')
        rendered_system_prompt = template.render(agents=agents_list)

        # Update the system message with the rendered content
        updated_messages = []
        system_message_updated = False
        for msg in state['messages']:
            if msg.get('role') == 'system' and not system_message_updated:
                updated_messages.append(
                    self.ollama_agent.create_message(
                        role='system',
                        content=rendered_system_prompt
                    )
                )
                system_message_updated = True
            else:
                updated_messages.append(msg)

        state['messages'] = updated_messages

        # Invoke Ollama with the updated messages
        try:
            messages_state: Messages = {'messages': state['messages']}
            result = await self.ollama_agent.invoke(state=messages_state)
            state['messages'] = result['messages']
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
    async def finalize_conversation(self, state: ContextState) -> ContextState:
        """
        Finalize the conversation and clean up resources.

        Provides a summary of active agents and logs completion.

        Parameters:
            state: Current context state.

        Returns:
            ContextState: Final state after cleanup and finalization.
        """
        self._log('Finalizing supervisor interaction.')

        # Log summary of active agents
        if state['agents']:
            self._log(f"Active agents at completion: {len(state['agents'])}")
            for agent_id, agent_info in state['agents'].items():
                self._log(
                    f"  Agent {agent_id}: {agent_info['query']} "
                    f"(Status: {agent_info['status']})"
                )
        else:
            self._log('No active agents.')

        self.messages_count = len(state['messages'])
        self._log(f'Total messages in conversation: {self.messages_count}')

        return state

    async def make_graph(self):
        """
        Build and compile the LangGraph workflow.

        Constructs a state graph with nodes for:
        1. Analyzing incoming tasks
        2. Processing supervisor decisions with tools
        3. Routing based on tool calls
        4. Finalizing the conversation

        Returns:
            None
        """
        # Initialize supervisor tools
        supervisor_tools = [
            {
                'name': 'create_agent',
                'description': 'Creates a new agent to handle a specific task.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The task description for the new agent.'
                        }
                    },
                    'required': ['query']
                },
                'tool_object': self.create_agent
            },
            {
                'name': 'delete_agent',
                'description': 'Deletes an existing agent by its ID.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'agent_id': {
                            'type': 'integer',
                            'description': 'The ID of the agent to delete.'
                        }
                    },
                    'required': ['agent_id']
                },
                'tool_object': self.delete_agent
            },
            {
                'name': 'skip_agent',
                'description': 'Skip agent management for this iteration.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                },
                'tool_object': self.skip_agent
            }
        ]

        # Retrieve tools from Ollama agent and merge with supervisor tools
        await self.ollama_agent.retrieve_tools(lang_tools=supervisor_tools)

        # Define the supervisor workflow
        workflow = StateGraph(
            ContextState,
            input_schema=InputState
        )

        # Add nodes
        workflow.add_node('set_initial_messages', self.set_initial_messages)
        workflow.add_node('analyze_task', self.analyze_task)
        workflow.add_node('finalize_conversation', self.finalize_conversation)

        # Add edges between nodes
        workflow.add_edge(START, 'set_initial_messages')
        workflow.add_edge('set_initial_messages', 'analyze_task')
        # After an agent step, check end conditions and proceed accordingly
        workflow.add_conditional_edges(
            'analyze_task',
            self.route_on_tool_call,
            {
                'agent': 'analyze_task',
                'finish': 'finalize_conversation'
            },
        )
        workflow.add_edge('finalize_conversation', END)

        # Compile the graph with memory persistence
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)  # type: ignore[assignment]
        self._log('Supervisor graph compiled successfully')
