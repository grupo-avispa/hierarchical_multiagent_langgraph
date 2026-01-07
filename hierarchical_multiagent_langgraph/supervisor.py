import asyncio
from typing import TypedDict
from pathlib import Path

from jinja2 import Template
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from ollama import Message

from hierarchical_multiagent_langgraph.agent import SinglePurposeAgent, AgentStatus

from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama
from langgraph_base_ros.chat_template_render import Messages



class InputState(TypedDict):
    """
    Represents the input state for the supervisor manager.

    Attributes:
        user_prompt: The user prompt for the conversation.
    """

    user_prompt: str


# class ContextState(Messages):
#     """
#     Represents the context state of a conversation with hierarchical agents.

#     This state extends the base Messages state to include metadata
#     about active agents being managed.

#     Attributes:
#         agents: Dictionary of active agents indexed by agent_id.
#         next_agent_id: Counter for assigning unique agent IDs.
#     """

#     agents: dict[int, dict]
#     next_agent_id: int


class SupervisorManager(LangGraphBase):

    def __init__(
        self,
        logger=None,
        ollama_agent: Ollama | None = None,
        max_steps: int = 5,
        system_prompt_path: str | None = None,
        spa_params: dict | None = None
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
            max_steps=max_steps
        )
        if self.ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')
        if spa_params is None:
            raise ValueError('spa_params must be provided to SupervisorManager.')
        self.spa_params = spa_params
        self.ollama_agent: Ollama = self.ollama_agent
        # Dictionary to store active agent instances
        self.active_agents: dict[int, SinglePurposeAgent] = {}
        # State for tracking agents (shared across tool calls)
        self.agents_state: dict = {'agents': {}, 'next_agent_id': 1}
        self._get_system_prompt(system_prompt_path)  # Load system prompt to attribute sys_prompt
        # Create tools with access to self
        self.supervisor_tools = self._create_supervisor_tools()

    # ========== LANGGRAPH TOOLS ==========

    def _create_supervisor_tools(self) -> list:
        """
        Create supervisor tools as closures with access to self.

        This method creates tools dynamically, allowing them to access
        instance attributes like ollama_agent, active_agents, and logger.

        Returns:
            list: List of tool dictionaries in the format expected by Ollama.
        """
        # Capture self in closure
        supervisor = self

        @tool(
            'create_agent',
            description='Creates a new agent to handle a specific task.'
        )
        def create_agent(query: str) -> str:
            """
            Create a new agent to handle the specified task.

            Parameters:
                query: The task description for the new agent.

            Returns:
                str: Confirmation message with the assigned agent ID.
            """
            agent_id = supervisor.agents_state['next_agent_id']

            supervisor._log(f'SUPERVISOR: Creating agent {agent_id} for task: {query}')

            # Prepare initial state for the agent
            initial_state : Messages = {
                "messages": [
                    Message(role="user", content=query)
                ]
            }
            # Extract parameters without modifying original dict
            spa_max_steps = supervisor.spa_params.get("max_steps", 5)
            spa_system_prompt_path = supervisor.spa_params.get("system_prompt_file")
            
            # Create a copy of spa_params without max_steps and system_prompt_file
            ollama_params = {k: v for k, v in supervisor.spa_params.items() 
                           if k not in ["max_steps", "system_prompt_file"]}

            # Create a new Ollama instance for this agent
            agent_ollama = Ollama(
                **ollama_params
            )
            
            # Create agent instance with its own Ollama instance
            new_agent = SinglePurposeAgent(
                logger=supervisor.logger,
                ollama_agent=agent_ollama,
                max_steps=spa_max_steps,
                system_prompt_path=spa_system_prompt_path
            )
            new_agent.set_id(agent_id)
            new_agent.set_status(AgentStatus.RUNNING)

            # Store agent info in internal state
            supervisor.agents_state['agents'][agent_id] = {
                'id': agent_id,
                'query': query,
                'status': AgentStatus.RUNNING.value
            }

            # Store agent instance
            supervisor.active_agents[agent_id] = new_agent

            # Increment agent counter
            supervisor.agents_state['next_agent_id'] += 1

            # Schedule agent execution in background without waiting
            # Use asyncio.ensure_future to schedule the coroutine without blocking
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(supervisor._run_agent(new_agent, initial_state))
                supervisor._log(f'SUPERVISOR: Agent {agent_id} scheduled for execution')
            except RuntimeError:
                # If no event loop, log warning but continue
                supervisor._log(f'WARNING: No event loop available to schedule agent {agent_id}')

            supervisor._log(f'SUPERVISOR: Agent {agent_id} created successfully')
            supervisor._log(f'SUPERVISOR: Current state: {supervisor.agents_state}')

            return f'Agent {agent_id} created successfully for task: {query}'

        @tool(
            'delete_agent',
            description='Deletes an existing agent by its ID.'
        )
        def delete_agent(agent_id: int) -> str:
            """
            Delete an existing agent by its ID.

            Parameters:
                agent_id: The ID of the agent to delete.

            Returns:
                str: Confirmation message.
            """
            if agent_id in supervisor.agents_state['agents']:
                supervisor._log(f'Deleting agent {agent_id}')
                agent_info = supervisor.agents_state['agents'][agent_id]
                del supervisor.agents_state['agents'][agent_id]
                if agent_id in supervisor.active_agents:
                    del supervisor.active_agents[agent_id]
                return f'Agent {agent_id} deleted successfully (was working on: {agent_info["query"]})'
            else:
                supervisor._log(f'Agent {agent_id} not found')
                return f'Error: Agent {agent_id} not found'

        @tool(
            'skip_agent',
            description='Skip agent management for this iteration.'
        )
        def skip_agent() -> str:
            """
            Skip agent management for this iteration.

            Returns:
                str: Confirmation message.
            """
            supervisor._log('Skipping agent action')
            return 'No agent action needed for this request'

        # Return tools in the format expected by Ollama
        return [
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
                'tool_object': create_agent
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
                'tool_object': delete_agent
            },
            {
                'name': 'skip_agent',
                'description': 'Skip agent management for this iteration.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                },
                'tool_object': skip_agent
            }
        ]

    async def _run_agent(self, agent: SinglePurposeAgent, 
                         initial_state: Messages) -> None:
        """
        Run an agent's graph asynchronously in the background.

        Parameters:
            agent: The agent instance to run.
            initial_state: Initial message state for the agent.

        Returns:
            None
        """
        agent_id = agent.get_id()
        try:
            self._log(f'AGENT {agent_id}: Starting execution pipeline...')
            
            # Ensure tools are registered before building the graph
            self._log(f'AGENT {agent_id}: Retrieving tools...')
            await agent.ollama_agent.retrieve_tools(agent.lang_tools)
            
            # Build the agent's graph if not already built
            if agent.graph is None:
                self._log(f'AGENT {agent_id}: Building graph...')
                await agent.make_graph()

            # Run the agent's graph
            self._log(f'AGENT {agent_id}: Executing task...')
            result = await agent.graph.ainvoke(initial_state)
            
            # Update agent status based on execution result
            final_status = agent.get_status()
            self._log(f'AGENT {agent_id}: Task completed with status: {final_status}')
            
            # Update internal state
            if agent_id in self.agents_state['agents']:
                self.agents_state['agents'][agent_id]['status'] = final_status

        except Exception as e:
            self._log(f'ERROR in AGENT {agent_id}: {e}')
            agent.set_status(AgentStatus.FAILURE)
            if agent_id in self.agents_state['agents']:
                self.agents_state['agents'][agent_id]['status'] = AgentStatus.FAILURE.value

    # ========== LANGGRAPH NODES ==========

    @traceable
    async def set_initial_messages(self, state: InputState) -> Messages:
        """
        Set initial messages from stored prompts and return the initial conversation state.

        This node processes the input state and creates the initial context state
        with system and user prompts as the first messages. The system prompt is
        rendered with Jinja2 using the initial empty agent context and user query.

        Parameters:
            state: The input state containing optional user prompt.

        Returns:
            Messages: The initialized conversation state with system and user messages.
        """
        # Extract user query from input state
        user_query = state.get('user_prompt', '')

        # Build context about current agents
        agents_list = [
            {
                'id': agent_id,
                'query': agent_info['query'],
                'status': agent_info['status']
            }
            for agent_id, agent_info in self.agents_state['agents'].items()
        ]
        
        # Render the system prompt with initial context (empty agents list)
        template = Template(self.sys_prompt)
        rendered_system_prompt = template.render(
            agents_context=agents_list
        )
        self._log(f'SUPERVISOR:\n--- Rendered system prompt  ---')
        self._log(f'\n\n{rendered_system_prompt}\n')
        self._log(f'\n------------------------------')

        # Create initial context state with rendered system prompt
        state: Messages = {
            'messages': [
                Message(
                    role='system',
                    content=rendered_system_prompt
                )
            ]
        }

        # Add user message if user prompt is set
        if user_query:
            state['messages'].append(
                Message(
                    role='user',
                    content=user_query
                )
            )

        return state

    @traceable
    async def analyze_task(self, state: Messages) -> Messages:
        """
        Analyze the incoming task and invoke the LLM to make supervisor decisions.

        This node:
        1. Re-renders the system prompt if agents context has changed.
        2. Calls the LLM with the current messages.

        Parameters:
            state: The current context state.

        Returns:
            Messages: Updated state with LLM response.
        """
        self._log('SUPERVISOR: Analyzing task and processing supervisor decision...')

        # Invoke Ollama with the current messages
        try:
            state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log(f'SUPERVISOR: Error during Ollama agent invocation: {e}')
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
        uc = 'agent'
        self._log(f'SUPERVISOR: Managing steps, current step: {self.steps}')
        # self._log(f'SUPERVISOR: Managing steps, current messages: {state["messages"]}')
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                # Finish if tool call detected
                self._log('Tool call detected in the last message.')
                uc = 'finish'
                if self.steps > self.max_steps:
                    self._log('Maximum steps reached, finishing interaction.')
            else:
                self._log('No tool call detected, trying again.')
                self._log(
                    'SUPERVISOR: Incorrect response from assistant:\n' + f"{state['messages'][-1]['content']}")
                state['messages'].append(
                    Message(
                        role='user',
                        content='Try again, remember to use the tools provided, you should not respond directly.'
                    )
                )
            # Update messages count
            self.messages_count = len(state['messages'])
            self._log(f'SUPERVISOR: Total messages in conversation: {self.messages_count}')
        except Exception as e:
            self._log(f'SUPERVISOR: Error in manage_steps: {e}')
            uc = 'finish'
        return uc

    @traceable
    async def finalize_conversation(self, state: Messages) -> Messages:
        """
        Finalize the conversation and clean up resources.

        Provides a summary of active agents and logs completion.

        Parameters:
            state: Current context state.

        Returns:
            Messages: Final state after cleanup and finalization.
        """
        self._log('SUPERVISOR: Finalizing supervisor interaction.')
        if self.steps >= self.max_steps:
            self._log('SUPERVISOR: Maximum steps reached during finalization.')
        else:
            self._log('SUPERVISOR: Agent reached final state before maximum steps.')
        self.steps = 0
        self.ollama_agent.reset_memory()

        # Log summary of active agents
        if self.agents_state['agents']:
            self._log(f"SUPERVISOR: Active agents at completion: {len(self.agents_state['agents'])}")
            for agent_id, agent_info in self.agents_state['agents'].items():
                self._log(
                    f"  Agent {agent_id}: {agent_info['query']} "
                    f"(Status: {agent_info['status']})"
                )
        else:
            self._log('No active agents.')

        self.messages_count = len(state['messages'])
        self._log(f'SUPERVISOR: FINAL STEP Total messages in conversation: {self.messages_count}')

        return state
    
    # ========== GRAPH GENERATION ==========

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

        # Define the supervisor workflow
        workflow = StateGraph(
            Messages,
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
