"""
Supervisor manager module for hierarchical multi-agent coordination using LangGraph.

This module provides the SupervisorManager class, which orchestrates multiple
SinglePurposeAgent instances in a hierarchical structure. The supervisor uses
LLM-based decision-making to create, manage, and coordinate the lifecycle of
agents based on user tasks and LLM reasoning.

Key components:
    - SupervisorManager: Main class that manages agent lifecycle and coordination.
    - AgentRegistry: Thread-safe registry for agent lifecycle management.
    - InputState: Input schema for the supervisor workflow.
"""

import asyncio
from typing import TypedDict
import traceback

from hierarchical_multiagent_langgraph.agent import AgentStatus, SinglePurposeAgent
from hierarchical_multiagent_langgraph.agent_executor import AgentExecutor
from hierarchical_multiagent_langgraph.agent_registry import (
    AgentRegistry,
    AgentTask,
    FinishedAgentsState,
    TaskPriority,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph_base_ros.chat_template_render import Messages
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama
from langsmith import traceable
from ollama import Message


class InputState(TypedDict):
    """
    Input schema for the supervisor manager workflow.

    Defines the initial state passed to the supervisor's LangGraph workflow.
    The user prompt is processed to create, coordinate, and manage multiple
    single-purpose agents.

    Attributes:
        user_prompt (str): The user's query or task description that the
            supervisor will decompose and delegate to appropriate agents.
    """

    user_prompt: str


class SupervisorManager(LangGraphBase):
    """
    Hierarchical multi-agent supervisor orchestrating task decomposition and agent coordination.

    The SupervisorManager employs an LLM-based supervisor agent to intelligently decompose
    complex user queries into subtasks and coordinate multiple SinglePurposeAgent instances
    for parallel execution. The supervisor maintains complete lifecycle management of agents
    from creation through result synthesis, while ensuring thread-safe concurrent access.

    Workflow:
        1. set_initial_messages: Initializes LangGraph state from user input.
        2. analyze_task: LLM supervisor analyzes the task and decides on agent actions.
        3. route_on_tool_call: Routes supervisor's tool calls
            (create_agent, delete_agent, skip_agent).
        4. finalize_conversation: Aggregates results from all agents and synthesizes response.

    Thread Safety:
        All agent lists (pending, running, finished) are protected by agent_lists_lock to
        ensure safe concurrent access from multiple agent execution threads.

    Attributes:
        registry (AgentRegistry): Thread-safe registry for agent lifecycle management.
        supervisor_tools (list): Tools exposed to LLM for agent lifecycle control.
        sys_prompt (str): System prompt guiding supervisor behavior and reasoning.
        spa_params (dict): Configuration parameters for SinglePurposeAgent instances.
        ollama_agent (Ollama): LLM instance for supervisor reasoning and planning.

    Raises:
        ValueError: If ollama_agent or spa_params are not provided during initialization.
    """

    def __init__(
        self,
        logger=None,
        ollama_agent: Ollama | None = None,
        max_steps: int = 5,
        system_prompt_path: str | None = None,
        spa_params: dict | None = None,
        agent_timeout: float = 120.0,
    ) -> None:
        """
        Initialize the Supervisor Manager.

        Sets up the supervisor's LangGraph workflow, tools, agent management structures,
        and loads the system prompt. Initializes thread-safe lists for tracking agents
        in various lifecycle states.

        Parameters:
            logger: Optional ROS2 logger for debug/info/warning output. If None,
                inherits from parent class. Defaults to None.
            ollama_agent (Ollama | None): Ollama LLM instance for supervisor
                reasoning. Required for operation. Defaults to None.
            max_steps (int): Maximum LangGraph execution steps before termination.
                Defaults to 5.
            system_prompt_path (str | None): Path to YAML/text file containing
                system prompt that guides supervisor behavior. Defaults to None.
            spa_params (dict | None): Configuration dictionary passed to all
                created SinglePurposeAgent instances. Required for operation.
                Defaults to None.
            agent_timeout (float): Maximum time in seconds for a single agent
                execution before it is forcefully terminated. Defaults to 120.0.

        Raises:
            ValueError: If ollama_agent is not provided.
            ValueError: If spa_params is not provided.
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
        # Thread-safe registry for agent lifecycle management
        self.registry = AgentRegistry()
        # Maximum time in seconds for a single agent execution
        self.agent_timeout = agent_timeout
        # Agent executor for running agents in background threads
        self.executor = AgentExecutor(
            registry=self.registry,
            agent_timeout=self.agent_timeout,
            logger=self.logger,
        )
        # Load system prompt to attribute sys_prompt
        self._get_system_prompt(system_prompt_path)
        # Create tools with access to self
        self.supervisor_tools = self._create_supervisor_tools()

    # ========== LANGGRAPH TOOLS ==========

    def _create_supervisor_tools(self) -> list:
        """
        Assemble supervisor tools for LLM-driven agent lifecycle management.

        Delegates to individual ``_build_*_tool()`` methods, then returns them
        in the dictionary format expected by the Ollama tool-calling API.

        Returns
        -------
        list
            Tool dictionaries with keys ``name``, ``description``,
            ``inputSchema``, and ``tool_object``.
        """
        create_agent_fn = self._build_create_agent_tool()
        delete_agent_fn = self._build_delete_agent_tool()
        skip_agent_fn = self._build_skip_agent_tool()

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
                        },
                        'priority': {
                            'type': 'string',
                            'description': (
                                'Execution priority level: high, medium, or low. '
                                'Higher priority agents are dispatched first.'
                            ),
                            'enum': ['high', 'medium', 'low'],
                            'default': 'medium'
                        }
                    },
                    'required': ['query']
                },
                'tool_object': create_agent_fn
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
                'tool_object': delete_agent_fn
            },
            {
                'name': 'skip_agent',
                'description': 'Skip agent management for this iteration.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                },
                'tool_object': skip_agent_fn
            }
        ]

    def _build_create_agent_tool(self):
        """
        Build the ``create_agent`` LangChain tool closure.

        Returns
        -------
        StructuredTool
            A ``@tool``-decorated callable that creates and enqueues a new
            ``SinglePurposeAgent``.
        """
        supervisor = self

        @tool(
            'create_agent',
            description='Creates a new agent to handle a specific task with a given priority.'
        )
        @traceable(name='sup_create_agent')
        def create_agent(query: str, priority: str = 'medium') -> str:
            """Create and queue a new SinglePurposeAgent for task execution.

            Parameters
            ----------
            query : str
                Task description / prompt for the new agent.
            priority : str
                Execution priority level. One of 'high', 'medium', or 'low'.
                Higher priority agents are dispatched first. Defaults to 'medium'.

            Returns
            -------
            str
                Confirmation message with the assigned agent ID.
            """
            # Map priority string to TaskPriority enum
            priority_map = {
                'high': TaskPriority.HIGH,
                'medium': TaskPriority.MEDIUM,
                'low': TaskPriority.LOW,
            }
            task_priority = priority_map.get(priority.lower(), TaskPriority.MEDIUM)
            agent_id = supervisor.registry.next_id()
            query = f'Your assigned agent ID is {agent_id}. And your task is: {query}'
            supervisor._log_info(f'SUPERVISOR: Creating agent {agent_id} for task: {query}')

            # Prepare initial state for the agent
            initial_state: Messages = {
                'messages': [
                    Message(role='user', content=query)
                ]
            }
            # Extract parameters without modifying original dict
            spa_max_steps = supervisor.spa_params.get('max_steps', 5)
            spa_system_prompt_path = supervisor.spa_params.get('system_prompt_file')

            # Create a copy of spa_params without max_steps and system_prompt_file
            ollama_params = {
                k: v for k, v in supervisor.spa_params.items()
                if k not in ['max_steps', 'system_prompt_file', 'mcp_servers_config']
            }
            # Create a new Ollama instance for this agent
            agent_ollama = Ollama(
                **ollama_params
            )
            # Create agent instance with its own Ollama instance
            new_agent = SinglePurposeAgent(
                logger=supervisor.logger,
                ollama_agent=agent_ollama,
                max_steps=spa_max_steps,
                system_prompt_path=spa_system_prompt_path,
                mcp_servers_config=supervisor.spa_params.get('mcp_servers_config')
            )
            new_agent.set_id(agent_id)

            # Add agent task to pending queue (producer-consumer pattern)
            agent_task = AgentTask(
                agent=new_agent,
                input_state=initial_state,
                priority=task_priority,
            )
            supervisor.registry.enqueue(agent_task)
            supervisor._log_info(
                f'SUPERVISOR: Agent {agent_id} (priority={priority}) '
                f'added to pending list successfully.')

            return (
                f'Agent {agent_id} created successfully '
                f'(priority={priority}) for task: {query}'
            )

        return create_agent

    def _build_delete_agent_tool(self):
        """
        Build the ``delete_agent`` LangChain tool closure.

        Returns
        -------
        StructuredTool
            A ``@tool``-decorated callable that cancels and removes a running
            agent by its ID.
        """
        supervisor = self

        @tool(
            'delete_agent',
            description='Deletes an existing agent by its ID.'
        )
        @traceable(name='sup_delete_agent')
        def delete_agent(agent_id: int) -> str:
            """Terminate and remove a running agent by its ID.

            Three-phase approach to avoid holding the registry lock during
            potentially slow network I/O (MCP ``stop_behavior_tree``).

            Parameters
            ----------
            agent_id : int
                Unique identifier of the agent to terminate.

            Returns
            -------
            str
                Status message indicating success or failure.
            """
            # Phase 1: Find the target agent (thread-safe lookup)
            target = supervisor.registry.find_running(agent_id)

            if target is None:
                message = f'Error: Agent {agent_id} not found in running agents'
                supervisor._log_info(message)
                return message

            query = target.input_prompt

            # Phase 2: Perform network I/O WITHOUT holding the lock
            if supervisor.ollama_agent.mcp_client is not None:
                try:
                    supervisor._log_info(
                        'Stopping behavior tree via MCP client...')
                    stop_future = asyncio.run_coroutine_threadsafe(
                        supervisor.ollama_agent.mcp_client.call_tool(
                            'stop_behavior_tree',
                            arguments={'execution_id': str(agent_id)}
                        ),
                        target.event_loop
                    )
                    result = stop_future.result(timeout=8.0)
                    supervisor._log_info(
                        f'Behavior tree stopped successfully. Result: {result}'
                    )
                except Exception as e:
                    supervisor._log_error(
                        f'ERROR stopping behavior tree for AGENT [{agent_id}]: '
                        f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
                    )

            # Cancel the agent's coroutine AFTER stopping the BT
            if target.event_loop is not None:
                target.event_loop.call_soon_threadsafe(
                    target.coroutine_handler.cancel
                )
            else:
                target.coroutine_handler.cancel()

            # Phase 3: Update registry (thread-safe mutation)
            supervisor.registry.remove_running(agent_id)
            finished_state = FinishedAgentsState(
                agent_id=agent_id,
                input_prompt=query,
                agent_result='Agent execution was cancelled by supervisor.',
                status=AgentStatus.FAILURE
            )
            supervisor.registry.add_finished(finished_state)

            message = (
                f'Agent {agent_id} deleted successfully '
                f'(was working on: {query})'
            )
            supervisor._log_info(message)
            return message

        return delete_agent

    def _build_skip_agent_tool(self):
        """
        Build the ``skip_agent`` LangChain tool closure.

        Returns
        -------
        StructuredTool
            A ``@tool``-decorated callable that performs no agent action.
        """
        supervisor = self

        @tool(
            'skip_agent',
            description='Skip agent management for this iteration.'
        )
        @traceable(name='sup_skip_agent')
        def skip_agent() -> str:
            """No-op tool: skip agent lifecycle management for current iteration.

            Returns
            -------
            str
                Confirmation message indicating no action was taken.
            """
            supervisor._log_info('Skipping agent action')
            return 'No agent action needed for this request'

        return skip_agent

    # ========== LANGGRAPH NODES ==========

    @traceable(name='sup_set_initial_messages')
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
        # Build context about current agents from registry
        agents_list = self.registry.get_agents_context()

        # Create initial context state with rendered system prompt
        sys_message = self._render_system_prompt(agents_context=agents_list)
        current_state: Messages = {
            'messages': [sys_message]
        }

        # Extract user query from input state and add to messages if present
        user_query = state.get('user_prompt', '')
        if user_query:
            current_state['messages'].append(
                Message(
                    role='user',
                    content=user_query
                )
            )

        return current_state

    @traceable(name='sup_analyze_task')
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
        self._log_info('SUPERVISOR: Analyzing task and processing supervisor decision...')

        # Invoke Ollama with the current messages
        try:
            state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log_error(f'SUPERVISOR: Error during Ollama agent invocation: {e}')
            raise

        return state

    @traceable(name='sup_route_on_tool_call')
    def route_on_tool_call(self, state: Messages) -> str:
        """
        Route the conversation flow based on tool call presence.

        Uses the shared ``_track_step()`` helper for step counting and tool call
        detection. The supervisor finishes when a tool call is detected (tool was
        executed successfully), and retries when no tool call is found (LLM
        responded with plain text instead of using tools).

        Parameters:
            state (Messages): Current conversation state with messages.

        Returns:
            str: Next node to transition to ('agent' to continue, 'finish' to end).
        """
        try:
            has_tool_call, max_steps_reached = self._track_step(state)
            self._log_info(f'SUPERVISOR: Managing steps, current step: {self.steps}')

            if max_steps_reached:
                self._log_warning('Maximum steps reached, finishing interaction NOW ...')
                return 'finish'

            if has_tool_call:
                self._log_info('Tool call detected in the last message.')
                return 'finish'

            # No tool call: ask LLM to retry with tools
            self._log_warning('No tool call detected, trying again.')
            self._log_warning(
                'SUPERVISOR: Incorrect response from assistant:\n'
                f"{state['messages'][-1]['content']}")
            state['messages'].append(
                Message(
                    role='user',
                    content='Try again, remember to use the tools provided, '
                    'you should not respond directly.'
                )
            )
            self._log_info(
                f'SUPERVISOR: Total messages in conversation: {self.messages_count}')
            return 'agent'
        except Exception as e:
            self._log_error(f'SUPERVISOR: Error in route_on_tool_call: {e}')
            return 'finish'

    @traceable(name='sup_finalize_conversation')
    async def finalize_conversation(self, state: Messages) -> Messages:
        """
        Finalize the conversation and clean up resources.

        Provides a summary of active agents and logs completion.

        Parameters:
            state: Current context state.

        Returns:
            Messages: Final state after cleanup and finalization.
        """
        self._log_info('SUPERVISOR: Finalizing supervisor interaction.')
        if self.steps >= self.max_steps:
            self._log_warning('SUPERVISOR: Maximum steps reached during finalization.')
        else:
            self._log_info('SUPERVISOR: Agent reached final state before maximum steps.')
        self.steps = 0
        self.ollama_agent.reset_memory()

        # Build context about current agents from registry snapshot
        summary = self.registry.get_summary()
        self._log_info('\n--- IDLE AGENTS ---\n')
        for agent_idle in summary['pending']:
            self._log_info(
                f'  Idle Agent [{agent_idle.agent.get_id()}]: '
                f'{agent_idle.input_state["messages"][0]["content"]} '
                f'(Status: IDLE)'
            )
        self._log_info('\n--- RUNNING AGENTS ---\n')
        for agent_run in summary['running']:
            self._log_info(
                f'  Running Agent [{agent_run.agent_id}]: {agent_run.input_prompt} '
                f'(Status: RUNNING)'
            )
        self._log_info('\n--- FINISHED AGENTS ---\n')
        for agent_finished in summary['finished']:
            self._log_info(
                f'  Finished Agent [{agent_finished.agent_id}]: '
                f'{agent_finished.input_prompt} '
                f'(Status: {agent_finished.status})'
            )
        self._log_info('\n-------------------\n')

        self.messages_count = len(state['messages'])
        self._log_info(
            f'SUPERVISOR: FINAL STEP Total messages in conversation: {self.messages_count}')
        # await self.current_task
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
        self._log_info('Supervisor graph compiled successfully')
