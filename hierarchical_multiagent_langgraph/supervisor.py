"""
Supervisor manager module for hierarchical multi-agent coordination using LangGraph.

This module provides the SupervisorManager class, which orchestrates multiple
SinglePurposeAgent instances in a hierarchical structure. The supervisor uses
LLM-based decision-making to create, manage, and coordinate the lifecycle of
agents based on user tasks and LLM reasoning.

Key components:
    - SupervisorManager: Main class that manages agent lifecycle and coordination.
    - AgentTask: Data structure for pending agent tasks.
    - RunningAgentsState: Tracks currently executing agents.
    - FinishedAgentsState: Tracks completed agent executions and results.
    - InputState: Input schema for the supervisor workflow.
"""

import asyncio
from dataclasses import dataclass
from threading import Lock
from typing import TypedDict

from hierarchical_multiagent_langgraph.agent import AgentStatus, SinglePurposeAgent
from jinja2 import Template
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


@dataclass
class AgentTask:
    """
    Encapsulates a pending agent task awaiting execution.

    This data structure represents a task that has been created by the supervisor's
    LLM but not yet started. Tasks are stored in a queue and executed asynchronously
    by the supervisor when resources become available.

    Attributes:
        agent (SinglePurposeAgent | None): The SinglePurposeAgent instance that
            will execute the task. Defaults to None (assigned during creation).
        input_state (Messages | None): The initial message state containing the
            task description and context for the agent. Defaults to None.
    """

    agent: SinglePurposeAgent = None  # type: ignore[assignment]
    input_state: Messages = None  # type: ignore[assignment]


@dataclass
class RunningAgentsState:
    """
    Tracks the execution state of a currently running agent.

    Maintains metadata about an agent that is actively executing its task.
    Enables monitoring, timeout management, and graceful cancellation of
    long-running agents. Stored in the supervisor's running agents list.

    Attributes:
        agent_id (int): Unique identifier of the running agent. Defaults to -1.
        input_prompt (str): The original input prompt/task description given
            to the agent. Defaults to empty string.
        coroutine_handler (asyncio.Task | None): The asyncio Task object that
            manages the agent's concurrent execution. Defaults to None.
        event_loop (asyncio.AbstractEventLoop | None): Reference to the event loop
            running the agent's coroutine. Used for cross-thread cancellation
            if needed. Defaults to None.
    """

    agent_id: int = -1
    input_prompt: str = ''
    coroutine_handler: asyncio.Task = None  # type: ignore[assignment]
    event_loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]


@dataclass
class FinishedAgentsState:
    """
    Captures the final state and results of a completed agent task.

    Contains the execution summary for an agent that has finished its task,
    whether successfully or with failure. This data is aggregated by the
    supervisor to synthesize final responses and provide feedback to the user.

    Attributes:
        agent_id (int): Unique identifier of the finished agent. Defaults to -1.
        input_prompt (str): The original task description given to the agent.
            Defaults to empty string.
        agent_result (str): The result or output produced by the agent's
            execution. Defaults to empty string.
        status (AgentStatus): The final execution status of the agent
            (SUCCESS, FAILURE, or IDLE). Defaults to AgentStatus.IDLE.
    """

    agent_id: int = -1
    input_prompt: str = ''
    agent_result: str = ''
    status: AgentStatus = AgentStatus.IDLE


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
        loop (asyncio.AbstractEventLoop): Event loop for async agent execution and management.
        pending_agents_list (list[AgentTask]): Queue of created but not-yet-started agents.
        running_agents_list (list[RunningAgentsState]): Agents currently executing their tasks.
        finished_agents_list (list[FinishedAgentsState]): Completed agents with results.
        agent_id_counter (int): Auto-incrementing counter for unique agent IDs.
        agent_lists_lock (Lock): Mutex protecting concurrent agent list access.
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
            loop (asyncio.AbstractEventLoop | None): Event loop for async agent
                execution. If None, uses asyncio.get_event_loop(). Defaults to None.

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
        # List for pending agent, running agent, and finished agent states
        self.pending_agents_list: list[AgentTask] = []
        self.running_agents_list: list[RunningAgentsState] = []
        self.finished_agents_list: list[FinishedAgentsState] = []
        self.agent_id_counter: int = 1
        # Mutex lock for thread-safe access to agent lists
        self.agent_lists_lock = Lock()
        # Load system prompt to attribute sys_prompt
        self._get_system_prompt(system_prompt_path)
        # Create tools with access to self
        self.supervisor_tools = self._create_supervisor_tools()

    # ========== LANGGRAPH TOOLS ==========

    def _create_supervisor_tools(self) -> list:
        """
        Create supervisor tools as closures with access to supervisor state.

        Dynamically creates three LLM tools (create_agent, delete_agent, skip_agent)
        that the supervisor agent can invoke for lifecycle management. Uses closure
        pattern to capture `self` reference, allowing tools to access and modify
        supervisor state like agent lists and counters.

        The tools are returned in Ollama's expected format with:
        - Tool name and description
        - JSON schema for input parameters
        - Callable function object

        Tool Lifecycle:
            1. create_agent: Instantiates new SinglePurposeAgent, adds to pending queue
            2. delete_agent: Cancels running agent gracefully via call_soon_threadsafe
            3. skip_agent: No-op action for supervisor iterations without agent changes

        Returns:
            list: List of tool dictionaries with keys: 'name', 'description',
                'inputSchema', 'tool_object'. Format required by Ollama LLM.

        Note:
            Tools use agent_lists_lock for thread-safe access to agent lists
            to prevent race conditions between supervisor and agent threads.
        """
        # Capture self in closure
        supervisor = self

        @tool(
            'create_agent',
            description='Creates a new agent to handle a specific task.'
        )
        def create_agent(query: str) -> str:
            """
            Create and queue a new SinglePurposeAgent for task execution.

            Instantiates a new agent with configuration from spa_params, assigns
            a unique ID, and adds it to the pending_agents_list for asynchronous
            execution. Uses producer-consumer pattern: supervisor produces agents,
            timer callback consumes and executes them in background threads.

            Agent Initialization:
                - Creates independent Ollama instance for the agent
                - Sets max_steps, system_prompt, and MCP config from spa_params
                - Initializes status to RUNNING
                - Generates unique auto-incremented agent ID

            Threading Model:
                - Runs synchronously in supervisor's event loop
                - Agent execution deferred to timer callback's thread
                - Uses agent_lists_lock for thread-safe queue insertion

            Parameters:
                query (str): Task description/prompt for the new agent.
                    Passed as initial message content.

            Returns:
                str: Confirmation message with assigned agent ID and task summary.
                    Format: "Agent {id} created successfully for task: {query}"
            """
            agent_id = supervisor.agent_id_counter
            supervisor.agent_id_counter += 1
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
            new_agent.set_status(AgentStatus.RUNNING)

            # Add agent task to pending queue (producer-consumer pattern)
            # The timer callback will consume and execute agents in their own threads
            agent_task = AgentTask(agent=new_agent, input_state=initial_state)
            supervisor.agent_lists_lock.acquire()
            supervisor.pending_agents_list.append(agent_task)
            supervisor.agent_lists_lock.release()
            supervisor._log_info(
                f'SUPERVISOR: Agent {agent_id} added to pending list successfully.')

            return f'Agent {agent_id} created successfully for task: {query}'

        @tool(
            'delete_agent',
            description='Deletes an existing agent by its ID.'
        )
        def delete_agent(agent_id: int) -> str:
            """
            Terminate and remove a running agent by its ID.

            Searches running_agents_list for the specified agent, cancels its
            execution gracefully using call_soon_threadsafe (for cross-thread safety),
            and removes it from the list. If agent not found, returns error message.

            Cancellation Strategy:
                - Uses asyncio's call_soon_threadsafe to safely cancel from supervisor thread
                - Handles both event_loop-aware and direct cancellation
                - Agent's asyncio.CancelledError handler stores final result before cleanup

            Thread Safety:
                - Acquires agent_lists_lock before list access
                - Cancellation is thread-safe across event loop boundaries
                - Running agent already removed from list before response

            Parameters:
                agent_id (int): Unique identifier of agent to terminate.

            Returns:
                str: Status message. Either:
                    - Success: "Agent {id} deleted successfully (was working on: {task})"
                    - Failure: "Error: Agent {id} not found in running agents"

            Side Effects:
                - Modifies running_agents_list (agent removed)
                - Cancels agent's coroutine in its event loop
                - Logs deletion action
            """
            # Initialize message
            message = f'Error: Agent {agent_id} not found in running agents'
            # Get lock for thread-safe access
            supervisor.agent_lists_lock.acquire()
            # Check if agent exists
            for _, running_agent in enumerate(supervisor.running_agents_list):
                if running_agent.agent_id == agent_id:
                    query = running_agent.input_prompt
                    # Use call_soon_threadsafe to cancel from another thread safely
                    if running_agent.event_loop is not None:
                        running_agent.event_loop.call_soon_threadsafe(
                            running_agent.coroutine_handler.cancel
                        )
                    else:
                        running_agent.coroutine_handler.cancel()
                    supervisor.running_agents_list.remove(running_agent)
                    message = f'Agent {agent_id} deleted successfully (was working on: {query})'
                    break
            supervisor.agent_lists_lock.release()

            supervisor._log_info(message)
            return message

        @tool(
            'skip_agent',
            description='Skip agent management for this iteration.'
        )
        def skip_agent() -> str:
            """
            No-op tool: skip agent lifecycle management for current iteration.

            Allows supervisor to acknowledge a workflow iteration without creating
            or modifying agents. Useful when supervisor decides task can be handled
            by existing agents or no new agents are needed at this time.

            Use Cases:
                - Monitoring phase: existing agents sufficient for task
                - Error recovery: wait before creating new agents
                - Resource constraints: defer agent creation to next iteration
                - Supervisor reasoning: task analysis requires no action

            Returns:
                str: Confirmation message indicating no action was taken.
            """
            supervisor._log_info('Skipping agent action')
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

    async def run_agent(
        self,
        agent: SinglePurposeAgent,
        initial_state: Messages
    ) -> None:
        """
        Execute a SinglePurposeAgent's LangGraph workflow asynchronously in background.

        Runs a complete agent execution pipeline: setup → tool retrieval → graph build →
        task execution → result collection. Handles all error cases gracefully, storing
        final results (success/failure) in finished_agents_list. Called by timer callback
        in dedicated thread context.

        Execution Pipeline:
            1. Setup Phase: Initialize MCP client if configured, verify connectivity
            2. Tool Setup: Retrieve tools available to agent from Ollama
            3. Graph Building: Construct agent's LangGraph workflow (once per agent)
            4. Execution: Run graph.ainvoke with initial state until completion
            5. Result Storage: Move from running→finished list with status/output
            6. Cleanup: Remove from running_agents_list, protect with agent_lists_lock

        State Transitions:
            - Created (pending) → Running (by timer) → Success/Failure (finished)
            - Cancellation: Running → Failure (CancelledError caught)
            - Exception: Running → Failure (any other exception)

        Setup Robustness:
            - MCP client failures logged but don't prevent execution
            - Tool retrieval optional; agent continues without unavailable tools
            - Allows graceful degradation if MCP servers unavailable

        Thread Safety:
            - Updates running_agents_list with agent_lists_lock protection
            - Updates finished_agents_list with agent_lists_lock protection
            - Safe to call from timer callback's thread context
            - Handles cross-thread cancellation via event_loop.call_soon_threadsafe()

        Parameters:
            agent (SinglePurposeAgent): Agent instance with configured LLM and tools.
            initial_state (Messages): Initial message state with user prompt/task.
                Format: {'messages': [Message(role='user', content='...')]}

        Returns:
            Messages: Final state from agent's graph execution (typically last message).

        Raises:
            asyncio.CancelledError: If supervisor calls delete_agent during execution.
                Caught internally; result stored before re-raising for cleanup.
            Exception: Other exceptions logged but caught; agent marked FAILURE.

        Side Effects:
            - Modifies agent status: RUNNING → SUCCESS/FAILURE
            - Moves agent from running_agents_list → finished_agents_list
            - Creates and executes agent's LangGraph (one-time per agent)
            - Logs detailed execution pipeline status
            - May initialize MCP client and retrieve tools
        """
        agent_id = agent.get_id()
        execution_result = FinishedAgentsState(
            agent_id=agent_id,
            input_prompt=initial_state['messages'][0]['content'],
            agent_result='Execution failed.',
            status=AgentStatus.FAILURE
        )
        # Try to setup the agent before execution.
        # If MCP client connection or tool retrieval fails,
        # we catch the exception and continue the agent execution without those features.
        try:
            self._log_info(f'AGENT [{agent_id}]: Starting execution pipeline...')
            # Ping MCP server to verify connection (already connected in create_agent)
            if agent.ollama_agent.mcp_client is not None:  # type: ignore[union-attr]
                async with agent.ollama_agent.mcp_client as client:  # type: ignore[union-attr]
                    await client.ping()
                self._log_info(f'AGENT [{agent_id}]: MCP client connection verified')

            # Ensure tools are registered before building the graph
            self._log_info(f'AGENT [{agent_id}]: Retrieving tools...')
            await agent.ollama_agent.retrieve_tools(agent.lang_tools)  # type: ignore[union-attr]

        except Exception as e:
            self._log_error(f'ERROR in AGENT [{agent_id}] during setup: {e}')

        # Execute the agent's graph and invoke its tasks
        try:
            # Build the agent's graph if not already built
            if agent.graph is None:
                self._log_info(f'AGENT [{agent_id}]: Building graph...')
                await agent.make_graph()

            # Run the agent's graph
            self._log_info(f'AGENT [{agent_id}]: Executing task...')
            result = await agent.graph.ainvoke(initial_state)  # type: ignore[attr-defined]
            execution_result.agent_result = result['messages'][-1]['content']

            # Update agent status based on execution result
            final_status = agent.get_status()
            execution_result.status = final_status
            self._log_info(
                f'AGENT [{agent_id}]: Task completed with status: {final_status}')

        except asyncio.CancelledError:
            self._log_error(f'AGENT [{agent_id}]: Execution cancelled by supervisor.')
            agent.set_status(AgentStatus.FAILURE)
            execution_result.status = AgentStatus.FAILURE
            execution_result.agent_result = 'Agent execution was cancelled.'
            # Store result before re-raising
            self.agent_lists_lock.acquire()
            # Agent already removed from running_agents_list by delete_agent
            self.finished_agents_list.append(execution_result)
            self.agent_lists_lock.release()
            # Re-raise to propagate cancellation to the event loop
            raise
        except Exception as e:
            self._log_error(f'ERROR in AGENT {agent_id}: {e}')
            agent.set_status(AgentStatus.FAILURE)
            execution_result.status = AgentStatus.FAILURE

        # Store the execution result in the finished agents list
        self.agent_lists_lock.acquire()
        for running_agent in self.running_agents_list:
            if running_agent.agent_id == agent_id:
                self.running_agents_list.remove(running_agent)
                break
        self.finished_agents_list.append(execution_result)
        self.agent_lists_lock.release()

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
        # Build context about current agents
        agents_list = [
            {
                'id': agent.agent_id,
                'query': agent.input_prompt,
                'result': '',
                'status': 'RUNNING'
            }
            for agent in self.running_agents_list
        ]
        agents_list.extend([
            {
                'id': agent.agent_id,
                'query': agent.input_prompt,
                'result': agent.agent_result,
                'status': agent.status
            }
            for agent in self.finished_agents_list
        ])

        # Render the system prompt with initial context (empty agents list)
        template = Template(self.sys_prompt)
        rendered_system_prompt = template.render(
            agents_context=agents_list
        )
        # self._log_info('SUPERVISOR:\n--- Rendered system prompt  ---')
        # self._log_info(f'\n\n{rendered_system_prompt}\n')
        # self._log_info('\n------------------------------')

        # Create initial context state with rendered system prompt
        current_state: Messages = {
            'messages': [
                Message(
                    role='system',
                    content=rendered_system_prompt
                )
            ]
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
        self._log_info('SUPERVISOR: Analyzing task and processing supervisor decision...')

        # Invoke Ollama with the current messages
        try:
            state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log_error(f'SUPERVISOR: Error during Ollama agent invocation: {e}')
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
        self.steps: int = self.steps + 1
        uc = 'agent'
        self._log_info(f'SUPERVISOR: Managing steps, current step: {self.steps}')
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                # Finish if tool call detected
                self._log_info('Tool call detected in the last message.')
                uc = 'finish'
            else:
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
            if self.steps > self.max_steps:
                self._log_warning('Maximum steps reached, finishing interaction NOW ...')
                uc = 'finish'
            # Update messages count
            self.messages_count = len(state['messages'])
            self._log_info(f'SUPERVISOR: Total messages in conversation: {self.messages_count}')
        except Exception as e:
            self._log_error(f'SUPERVISOR: Error in manage_steps: {e}')
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
        self._log_info('SUPERVISOR: Finalizing supervisor interaction.')
        if self.steps >= self.max_steps:
            self._log_warning('SUPERVISOR: Maximum steps reached during finalization.')
        else:
            self._log_info('SUPERVISOR: Agent reached final state before maximum steps.')
        self.steps = 0
        self.ollama_agent.reset_memory()

        # Build context about current agents
        self.agent_lists_lock.acquire()
        # Log pending agents
        self._log_info('\n--- IDLE AGENTS ---\n')
        for agent_idle in self.pending_agents_list:
            self._log_info(
                f'  Idle Agent [{agent_idle.agent.get_id()}]: '
                f'{agent_idle.input_state["messages"][0]["content"]} '
                f'(Status: IDLE)'
            )
        # Log running agents
        self._log_info('\n--- RUNNING AGENTS ---\n')
        for agent_run in self.running_agents_list:
            self._log_info(
                f'  Running Agent [{agent_run.agent_id}]: {agent_run.input_prompt} '
                f'(Status: RUNNING)'
            )
        # Log finished agents
        self._log_info('\n--- FINISHED AGENTS ---\n')
        for agent_finished in self.finished_agents_list:
            self._log_info(
                f'  Finished Agent [{agent_finished.agent_id}]: {agent_finished.input_prompt} '
                f'(Status: {agent_finished.status})'
            )

        self.agent_lists_lock.release()

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
