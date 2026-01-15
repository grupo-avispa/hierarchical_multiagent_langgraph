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
    Represents the input state for the supervisor manager.

    Attributes:
        user_prompt: The user prompt for the conversation.
    """

    user_prompt: str


@dataclass
class AgentTask:
    """
    Represents a pending agent task to be executed.

    Attributes:
        agent: The SinglePurposeAgent instance to execute.
        input_state: The initial message for the agent.
    """

    agent: SinglePurposeAgent = None  # type: ignore[assignment]
    input_state: Messages = None


@dataclass
class RunningAgentsState:
    """
    Represents the state of running agents.

    Attributes:
        agent_id: ID of the running agent.
        input_prompt: The input prompt used by the agent.
        coroutine_handler: The asyncio task handling the agent execution.
        event_loop: The asyncio event loop running the task (for cross-thread cancel).
    """

    agent_id: int = -1
    input_prompt: str = ''
    coroutine_handler: asyncio.Task = None  # type: ignore[assignment]
    event_loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]


@dataclass
class FinishedAgentsState:
    """
    Represents the state of finished agents.

    Attributes:
        agent_ids: ID of the finished agent.
        input_prompt: The input prompt used by the agent.
        agent_result: The result string from the agent.
        status: The final status of the agent.
    """

    agent_id: int = -1
    input_prompt: str = ''
    agent_result: str = ''
    status: AgentStatus = AgentStatus.IDLE


class SupervisorManager(LangGraphBase):
    """
    Hierarchical multi-agent supervisor that coordinates multiple SinglePurposeAgent instances.

    The SupervisorManager uses LLM-based decision-making to manage the lifecycle of agents,
    including creating new agents for specific tasks, monitoring their execution, and
    aggregating results. It maintains thread-safe lists of pending, running, and finished
    agents, and provides tools for the LLM to interact with the agent management system.

    The supervisor implements a LangGraph workflow that:
    1. Receives user prompts and initializes the conversation state.
    2. Analyzes tasks and decides on agent creation/deletion based on LLM reasoning.
    3. Routes decisions through tool calls (create_agent, delete_agent, skip_agent).
    4. Finalizes the conversation with aggregated results from all agents.

    Attributes:
        loop (asyncio.AbstractEventLoop): The event loop for managing async agent execution.
        pending_agents_list (list[AgentTask]): Queue of agents waiting to be executed.
        running_agents_list (list[RunningAgentsState]): List of currently executing agents.
        finished_agents_list (list[FinishedAgentsState]): List of completed agents with results.
        agent_id_counter (int): Counter for generating unique agent IDs.
        agent_lists_lock (Lock): Thread-safe mutex for agent list access.
        supervisor_tools (list): List of available tools for the LLM supervisor.
        sys_prompt (str): System prompt loaded from file for supervisor behavior.
        spa_params (dict): Parameters passed to SinglePurposeAgent instances.
        ollama_agent (Ollama): Ollama LLM instance for agent reasoning.
    """

    def __init__(
        self,
        logger=None,
        ollama_agent: Ollama | None = None,
        max_steps: int = 5,
        system_prompt_path: str | None = None,
        spa_params: dict | None = None,
        loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        """
        Initialize the Supervisor Manager.

        Creates the LLM instance with default configuration.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).
            ollama_agent: Optional Ollama agent instance (default: None).
            max_steps: Maximum number of steps for the graph execution (default: 5).
            system_prompt: The system prompt for the supervisor.
            spa_params: Parameters for SinglePurposeAgent instances.
            loop: Optional asyncio event loop to use (default: None).

        Returns:
            None
        """
        super().__init__(
            logger=logger,
            ollama_agent=ollama_agent,
            max_steps=max_steps
        )
        self.loop = loop if loop is not None else asyncio.get_event_loop()
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
        Create supervisor tools as closures with access to self.

        This method creates tools dynamically, allowing them to access
        instance attributes like ollama_agent or logger.

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
            agent_id = supervisor.agent_id_counter
            supervisor.agent_id_counter += 1

            supervisor._log(f'SUPERVISOR: Creating agent {agent_id} for task: {query}')

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
            supervisor._log(f'SUPERVISOR: Agent {agent_id} added to pending list successfully.')

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

            supervisor._log(message)
            return message

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

    async def run_agent(
        self,
        agent: SinglePurposeAgent,
        initial_state: Messages
    ) -> Messages:
        """
        Run an agent's graph asynchronously in the background.

        Parameters:
            agent: The agent instance to run.
            initial_state: Initial message state for the agent.

        Returns:
            Messages: The final state after agent execution.
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
            self._log(f'AGENT [{agent_id}] {agent_id}: Starting execution pipeline...')
            # Ping MCP server to verify connection (already connected in create_agent)
            if agent.ollama_agent.mcp_client is not None:
                async with agent.ollama_agent.mcp_client as client:
                    self._log(f'AGENT [{agent_id}] {agent_id}: PING ...')
                    await client.ping()
                self._log(f'AGENT [{agent_id}] {agent_id}: MCP client connection verified')

            # Ensure tools are registered before building the graph
            self._log(f'AGENT [{agent_id}] {agent_id}: Retrieving tools...')
            await agent.ollama_agent.retrieve_tools(agent.lang_tools)

        except Exception as e:
            self._log(f'ERROR in AGENT {agent_id} during setup: {e}')

        # Execute the agent's graph and invoke its tasks
        try:
            # Build the agent's graph if not already built
            if agent.graph is None:
                self._log(f'AGENT [{agent_id}] {agent_id}: Building graph...')
                await agent.make_graph()

            # Run the agent's graph
            self._log(f'AGENT [{agent_id}] {agent_id}: Executing task...')
            result = await agent.graph.ainvoke(initial_state)
            execution_result.agent_result = result['messages'][-1]['content']

            # Update agent status based on execution result
            final_status = agent.get_status()
            execution_result.status = final_status
            self._log(f'AGENT [{agent_id}] {agent_id}: Task completed with status: {final_status}')

        except asyncio.CancelledError:
            self._log(f'AGENT [{agent_id}]: Execution cancelled by supervisor.')
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
            self._log(f'ERROR in AGENT {agent_id}: {e}')
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
        self._log('SUPERVISOR:\n--- Rendered system prompt  ---')
        self._log(f'\n\n{rendered_system_prompt}\n')
        self._log('\n------------------------------')

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
        self.steps: int = self.steps + 1
        uc = 'agent'
        self._log(f'SUPERVISOR: Managing steps, current step: {self.steps}')
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                # Finish if tool call detected
                self._log('Tool call detected in the last message.')
                uc = 'finish'
            else:
                self._log('No tool call detected, trying again.')
                self._log(
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
                self._log('Maximum steps reached, finishing interaction NOW ...')
                uc = 'finish'
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

        # Build context about current agents
        self.agent_lists_lock.acquire()
        # Log pending agents
        self._log('\n--- IDLE AGENTS ---\n')
        for agent_idle in self.pending_agents_list:
            self._log(
                f'  Idle Agent [{agent_idle.agent.get_id()}]: '
                f'{agent_idle.input_state["messages"][0]["content"]} '
                f'(Status: IDLE)'
            )
        # Log running agents
        self._log('\n--- RUNNING AGENTS ---\n')
        for agent_run in self.running_agents_list:
            self._log(
                f'  Running Agent [{agent_run.agent_id}]: {agent_run.input_prompt} '
                f'(Status: RUNNING)'
            )
        # Log finished agents
        self._log('\n--- FINISHED AGENTS ---\n')
        for agent_finished in self.finished_agents_list:
            self._log(
                f'  Finished Agent [{agent_finished.agent_id}]: {agent_finished.input_prompt} '
                f'(Status: {agent_finished.status})'
            )

        self.agent_lists_lock.release()

        self.messages_count = len(state['messages'])
        self._log(f'SUPERVISOR: FINAL STEP Total messages in conversation: {self.messages_count}')
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
        self._log('Supervisor graph compiled successfully')
