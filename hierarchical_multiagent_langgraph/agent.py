"""
Single-purpose agent module for hierarchical multi-agent system using LangGraph.

This module provides the SinglePurposeAgent class, which represents a specialized
agent that focuses on a single task or query. Each agent has its own LangGraph
workflow for processing tasks, accessing tools, and interacting with an LLM backend.

Key components:
    - SinglePurposeAgent: An agent instance that executes a specific task.
    - AgentStatus: Enumeration of possible agent states (IDLE, RUNNING, SUCCESS, FAILURE).

The agent supports:
    - LLM-based reasoning through Ollama integration.
    - Model Context Protocol (MCP) servers for extended tool capabilities.
    - Tool execution and management via LangGraph workflows.
    - Status tracking and result aggregation.
"""

from enum import Enum
import json

from fastmcp import Client
from langgraph.graph import START, StateGraph
from langgraph_base_ros.chat_template_render import Messages
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama
from langsmith import traceable


class AgentStatus(str, Enum):
    """
    Enumeration of possible agent execution states.

    Represents the lifecycle status of a SinglePurposeAgent during task execution.
    Allows the supervisor to track agent progress and determine next actions.

    Attributes
    ----------
    IDLE : str
        Agent has been created but has not yet started execution.
    RUNNING : str
        Agent is currently executing its assigned task.
    SUCCESS : str
        Agent successfully completed its task.
    FAILURE : str
        Agent encountered an error and could not complete the task.

    """

    IDLE = 'idle'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class SinglePurposeAgent(LangGraphBase):
    """
    A specialized agent that executes a single task or query using LangGraph workflows.

    The SinglePurposeAgent is designed to handle a focused, well-defined task assigned
    by a supervisor. It maintains its own state machine with LangGraph nodes for
    reasoning, tool execution, and task completion. Each agent has a unique ID, query,
    and status tracking for integration with hierarchical multi-agent systems.

    Workflow:
        1. initialize_messages: Sets up initial message state from the assigned task.
        2. run_agent: Executes the main reasoning loop with LLM-based decision-making.
        3. Tool execution: Calls available tools (local or MCP-based) to perform work.
        4. finalize: Synthesizes results and transitions to SUCCESS or FAILURE state.

    Tool Integration:
        Supports both direct Python tools via langchain and extended tool capabilities
        through Model Context Protocol (MCP) servers. Tools are loaded dynamically from
        configuration files at initialization.

    LLM Integration:
        Uses Ollama LLM for agent reasoning, planning, and decision-making throughout
        task execution. Respects max_steps limit to prevent infinite loops.

    Attributes
    ----------
    id : int
        Unique identifier assigned by supervisor. Initialized to -1.
    status : AgentStatus
        Current execution state (IDLE, RUNNING, SUCCESS, FAILURE).
    lang_tools : list
        LangChain-style tools available to the agent, in addition to any
        MCP-based tools from mcp_client. Currently always empty: nothing in
        this class populates it (see the ``LANGGRAPH TOOLS`` section below).
    sys_prompt : str
        System prompt from file guiding agent behavior and reasoning.
    ollama_agent : Ollama
        LLM instance for agent task reasoning and execution.
    mcp_client : Client | None
        Optional MCP client providing extended tool access.
    max_steps : int
        Maximum LangGraph steps before task termination.
    logger
        Optional ROS2 logger inherited from parent LangGraphBase.

    Raises
    ------
    ValueError
        If ollama_agent is not provided during initialization.

    """

    def __init__(
            self,
            logger=None,
            ollama_agent: Ollama | None = None,
            max_steps: int = 5,
            system_prompt_path: str | None = None,
            mcp_servers_config: str | None = None
    ) -> None:
        """
        Initialize a SinglePurposeAgent instance.

        Sets up the agent with LLM configuration, loads system prompt and tools,
        and initializes MCP client if provided. The agent is ready for task
        assignment after initialization.

        Parameters
        ----------
        logger
            Optional ROS2 logger for debug/info output. If None,
            inherits from parent class. Defaults to None.
        ollama_agent : Ollama | None
            Ollama LLM instance for agent
            reasoning and task execution. Required. Defaults to None.
        max_steps : int
            Maximum LangGraph execution steps before task
            termination. Prevents infinite loops. Defaults to 5.
        system_prompt_path : str | None
            Path to YAML/text file containing
            system prompt that guides agent behavior. Defaults to None.
        mcp_servers_config : str | None
            Path to JSON file with Model Context
            Protocol server configurations for extended tool access. If provided,
            MCP client is initialized; if initialization fails, continues without
            it. Defaults to None.

        Raises
        ------
        ValueError
            If ollama_agent is not provided.

        Note:
            MCP client initialization failures are logged but do not prevent
            agent initialization. The agent will function with available tools only.

        """
        if ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')

        super().__init__(
            logger=logger,
            ollama_agent=ollama_agent,
            max_steps=max_steps,
        )

        # Unique identifier for the agent
        self.id: int = -1
        # Current status of the agent
        self.status: AgentStatus = AgentStatus.IDLE
        # List of tools available to the agent
        self.lang_tools: list = []
        # Load system prompt to attribute sys_prompt
        self._get_system_prompt(system_prompt_path)
        # Initialize MCP client if configuration is provided
        if mcp_servers_config is not None:
            try:
                # Open json config file
                with open(mcp_servers_config, 'r') as f:
                    config_data = json.load(f)
                self.ollama_agent.mcp_client = Client(config_data)  # type: ignore[union-attr]
                self._log_info(f'AGENT [{self.id}]: MCP client initialized successfully')
            except Exception as e:
                self._log_info(f'AGENT [{self.id}]: Error initializing MCP client: {e}')
                self.ollama_agent.mcp_client = None  # type: ignore[union-attr]

    def set_id(self, agent_id: int) -> None:
        """
        Assign a unique identifier to the agent.

        Called by supervisor during agent creation to set auto-generated ID.
        ID is used for tracking, logging, and deletion requests.

        Parameters
        ----------
        agent_id : int
            Unique auto-incremented identifier assigned by supervisor.

        Returns
        -------
        None

        """
        self.id = agent_id

    def get_id(self) -> int:
        """
        Retrieve the agent's unique identifier.

        Returns
        -------
        int
            The agent's unique ID assigned during creation, or -1 if not yet assigned.

        """
        return self.id

    def get_status(self) -> AgentStatus:
        """
        Retrieve the current execution status of the agent.

        Returns
        -------
        AgentStatus
            Current status (IDLE, RUNNING, SUCCESS, or FAILURE).
            Transitions from RUNNING→SUCCESS/FAILURE during execution.

        """
        return self.status

    def set_status(self, status: AgentStatus) -> None:
        """
        Update the agent's execution status.

        Called during initialization (RUNNING) and finalization (SUCCESS/FAILURE).
        Allows supervisor to track agent lifecycle.

        Parameters
        ----------
        status : AgentStatus
            New status to assign (IDLE, RUNNING, SUCCESS, FAILURE).

        Returns
        -------
        None

        """
        self.status = status

    # ========== LANGGRAPH NODES ==========

    @traceable(name='spa_query_response')
    async def query_response(self, state: Messages) -> Messages:
        """
        Generate LLM reasoning step and update conversation state with response.

        Core node of agent's reasoning loop. Invokes Ollama LLM to process current
        messages and generate next step (tool call or final response). Handles system
        prompt injection with Jinja2 templating and optional MCP resource embedding.

        System Prompt Handling:
            - Detects if system message already present in state
            - Retrieves MCP resources if MCP client configured
            - Renders system prompt template with available resources
            - Prepends system message to beginning of message list

        MCP Integration:
            - Queries MCP servers for available resources (if client initialized)
            - Embeds resource content into system prompt for context
            - Gracefully continues if MCP resource retrieval fails
            - Allows agent to access external knowledge sources

        State Management:
            - Sets agent status to RUNNING
            - Invokes ollama_agent.invoke() to get LLM response
            - Stores updated state with new LLM message
            - Preserves conversation history

        Parameters
        ----------
        state : Messages
            Current conversation state with message history.
            Format: {'messages': [Message(role, content), ...]}

        Returns
        -------
        Messages
            Updated state with new message(s) from LLM response.
            LLM may add tool calls or final response message.

        Raises
        ------
        ValueError
            If Ollama agent invocation fails (logged and re-raised).

        Side Effects:
            - Sets self.status = AgentStatus.RUNNING
            - Updates self.state with new agent response
            - Modifies state['messages'] in-place (prepends system message if needed)
            - Connects to MCP servers if configured

        """
        self.status = AgentStatus.RUNNING
        # Invoke Ollama agent
        try:
            # Check if any of the message roles is 'system'
            has_sys_message = any(
                (msg['role'] == 'system' and msg['content'] is not None)
                for msg in state['messages'])
            if not has_sys_message:
                # Get resources
                resources_content = []
                if self.ollama_agent is not None and self.ollama_agent.mcp_client is not None:
                    try:
                        async with self.ollama_agent.mcp_client as client:
                            resources = await client.list_resources()
                            for resource in resources:
                                content = await client.read_resource(resource.uri)
                                resources_content.append(
                                    content[0].text)
                    except Exception as e:
                        self._log_warning(f'Error retrieving MCP tools: {e}')
                # Prepend rendered system prompt
                sys_message = self._render_system_prompt(
                    resources=resources_content)
                state['messages'].insert(0, sys_message)
            self.state = await self.ollama_agent.invoke(state=state)  # type: ignore[union-attr]
        except ValueError as e:
            self._log_error(f'AGENT [{self.id}]: Error during Ollama agent invocation: {e}')
            raise

        return self.state

    @traceable(name='spa_manage_steps')
    def manage_steps(self, state: Messages) -> str:
        """
        Conditional routing: determine whether to continue reasoning or finish task.

        Examines the last message in conversation to decide next action. Uses
        the shared ``_track_step()`` helper for step counting and tool call
        detection. Routes to either another query_response cycle or finalization.

        Routing Logic:
            - Tool call present and under max_steps → 'agent' (continue)
            - Tool call present but max_steps reached → 'finish' (force exit)
            - No tool call → 'finish' (task naturally complete)

        Parameters
        ----------
        state : Messages
            Current conversation state with message history.

        Returns
        -------
        str
            Next node identifier ('agent' or 'finish') for conditional edge routing.

        """
        try:
            has_tool_call, max_steps_reached = self._track_step(state)
            self._log_info(f'AGENT [{self.id}]: Managing steps, current step: {self.steps}')

            if max_steps_reached:
                self._log_info(f'AGENT [{self.id}]: Maximum steps reached, finishing interaction.')
                return 'finish'

            if has_tool_call:
                return 'agent'

            self._log_info(f'AGENT [{self.id}]: No tool call detected, finishing interaction.')
            self._log_info(
                f'AGENT [{self.id}]: Final response from assistant:\n'
                f"{state['messages'][-1]['content']}"
            )
            self._log_info(
                f'AGENT [{self.id}]: Total messages in conversation: {self.messages_count}')
            return 'finish'
        except Exception as e:
            self._log_warning(f'AGENT [{self.id}]: Error in manage_steps: {e}')
            return 'finish'

    @traceable(name='spa_finish_ollama_interaction')
    async def finish_ollama_interaction(self, state: Messages) -> Messages:
        """
        Finalize agent task execution and transition to terminal state.

        Called when agent reaches completion (no more tool calls) or max_steps exceeded.
        Updates agent status based on execution completeness, resets step counter,
        and clears LLM memory. Marks end of single-purpose task execution.

        Status Logic:
            - SUCCESS: Agent completed task before reaching max_steps
            - FAILURE: Agent reached max_steps limit without natural completion
            - Both: Will be collected by supervisor for result synthesis

        Memory Management:
            - Clears ollama_agent.memory to free LLM conversation history
            - Resets self.steps counter to 0 for next potential use
            - Preserves final state with complete message history

        State Return:
            - Returns unmodified input state for supervisor collection
            - Final message contains agent's last response/tool call
            - Supervisor extracts agent_result from last message content

        Parameters
        ----------
        state : Messages
            Final conversation state at completion.
            Contains all messages from task start to finish.

        Returns
        -------
        Messages
            Unmodified final state returned to supervisor/caller.

        Side Effects:
            - Sets self.status = SUCCESS or FAILURE based on max_steps check
            - Resets self.steps = 0
            - Calls ollama_agent.reset_memory() to clear LLM state
            - Logs finalization status

        """
        self._log_info(f'AGENT [{self.id}]: Finalizing Ollama interaction.')
        if self.steps >= self.max_steps:
            self._log_info(f'AGENT [{self.id}]: Maximum steps reached during finalization.')
            self.status = AgentStatus.FAILURE
        else:
            self._log_info(f'AGENT [{self.id}]: Agent reached final state before maximum steps.')
            self.status = AgentStatus.SUCCESS
        self.steps = 0
        self.ollama_agent.reset_memory()  # type: ignore[union-attr]
        return state

    # ========== GRAPH GENERATION ==========

    async def make_graph(self):
        """
        Construct and compile the agent's LangGraph state machine workflow.

        Builds a directed acyclic graph (DAG) with two nodes and conditional routing:
        1. query_response: Main LLM reasoning node (async)
        2. finish_ollama_interaction: Finalization node (async)

        Graph Flow:
            START → query_response ↓ (conditional)
                ├─ 'agent' → query_response (loop for tool calls)
                └─ 'finish' → finish_ollama_interaction → END

        Conditional Routing:
            The manage_steps() method decides routing:
            - Returns 'agent': More tool calls needed, loop back to query_response
            - Returns 'finish': Task complete or max_steps reached, proceed to finalization

        Execution Model:
            - Compiled graph supports async execution (ainvoke)
            - Maintains conversation state across nodes
            - Enforces step limits to prevent infinite loops
            - Supports tool calls within Ollama integration

        Stored Result:
            - Compiled graph stored in self.graph
            - Ready for invocation by supervisor/run_agent()
            - One graph per agent instance

        Uses instance attributes (query_response, manage_steps, etc.) rather
        than explicit parameters.

        Returns
        -------
        None
            Compiled graph stored in self.graph attribute.

        Side Effects:
            - Creates self.graph (compiled StateGraph)
            - Ready for async execution via graph.ainvoke(initial_state)

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

    # Additional LangChain-style tools (decorated with @tool) can be added
    # here for the agent to call during its reasoning loop, alongside the
    # MCP-based tools already available through mcp_client.
