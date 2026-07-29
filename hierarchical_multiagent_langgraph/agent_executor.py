# Copyright (c) 2026 Alberto Tudela
# Copyright (c) 2026 Grupo Avispa, DTE, Universidad de Málaga
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agent executor module for managing async execution of single-purpose agents.

This module provides the AgentExecutor class, which handles the complete
execution pipeline for SinglePurposeAgent instances: setup, tool retrieval,
graph building, task execution, and result collection. A bounded pool of
worker threads pulls tasks from the registry's priority queue, each worker
reusing its own asyncio event loop across the agents it processes in
sequence.

Key components:
    - AgentExecutor: Manages async execution lifecycle for agents.
"""

import asyncio
import logging
import threading

from hierarchical_multiagent_langgraph.agent import AgentStatus, SinglePurposeAgent
from hierarchical_multiagent_langgraph.agent_registry import (
    AgentRegistry,
    FinishedAgentsState,
    RunningAgentsState,
)
from langgraph_base_ros.chat_template_render import Messages


class AgentExecutor:
    """
    Manage async execution of SinglePurposeAgent instances in a bounded pool.

    A fixed number of persistent worker threads (``max_concurrent_agents``)
    each pull the next task from the registry's priority queue as soon as
    they are free, blocking with no polling delay when no work is pending.
    Because all workers draw from the same priority queue, a ``high``
    priority agent created while ``max_concurrent_agents`` other agents are
    already running gets dispatched as soon as any worker frees up, ahead of
    any ``medium``/``low`` priority task still waiting -- unlike the
    previous design, which spawned an unbounded thread per agent and made
    priority meaningless once every pending task started at once.

    Attributes
    ----------
    registry : AgentRegistry
        Thread-safe registry for agent lifecycle state management.
    agent_timeout : float
        Maximum time in seconds for a single agent execution.
    max_concurrent_agents : int
        Maximum number of agents executing at the same time.
    logger : logging.Logger | None
        Optional logger instance for debug/info/warning output.

    """

    def __init__(
        self,
        registry: AgentRegistry,
        agent_timeout: float = 120.0,
        max_concurrent_agents: int = 4,
        logger=None,
    ) -> None:
        """
        Initialize the AgentExecutor.

        Parameters
        ----------
        registry : AgentRegistry
            Thread-safe registry for tracking agent lifecycle states.
        agent_timeout : float
            Maximum time in seconds for a single agent execution
            before it is forcefully terminated. Defaults to 120.0.
        max_concurrent_agents : int
            Maximum number of agents executing at the same time.
            Defaults to 4.
        logger : logging.Logger | None
            Optional logger for debug/info/warning output. If None,
            uses Python's standard logging module. Defaults to None.

        """
        self.registry = registry
        self.agent_timeout = agent_timeout
        self.max_concurrent_agents = max_concurrent_agents
        self.logger = logger
        # Persistent worker thread references
        self._worker_threads: list[threading.Thread] = []

    def _log_info(self, msg: str) -> None:
        """Log info message using provided logger or Python logging."""
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

    def _log_error(self, msg: str) -> None:
        """Log error message using provided logger or Python logging."""
        if self.logger is not None:
            self.logger.error(msg)
        else:
            logging.error(msg)

    async def run_agent(
        self,
        agent: SinglePurposeAgent,
        initial_state: Messages,
    ) -> None:
        """
        Execute a SinglePurposeAgent's LangGraph workflow asynchronously.

        Run a complete agent execution pipeline: setup, tool retrieval,
        graph build, task execution, and result collection. Handle all error
        cases gracefully, storing final results (success/failure) in the
        registry's finished list.

        Parameters
        ----------
        agent : SinglePurposeAgent
            Agent instance with configured LLM and tools.
        initial_state : Messages
            Initial message state with user prompt/task.

        Raises
        ------
        asyncio.CancelledError
            If the supervisor calls delete_agent during execution. Caught
            internally; result stored before re-raising for cleanup.

        Notes
        -----
        The MCP connection is opened once here and kept open for the whole
        execution instead of reconnecting on every ping/resource-list/tool
        call. fastmcp's ``Client`` context manager is reentrant (internal
        nesting counter), so the ``async with`` calls elsewhere in the
        codebase (``query_response``'s resource listing,
        ``Ollama.invoke``'s tool execution) reuse this same session instead
        of paying a new connection handshake each time.

        """
        agent_id = agent.get_id()
        execution_result = FinishedAgentsState(
            agent_id=agent_id,
            input_prompt=initial_state['messages'][0]['content'],
            agent_result='Execution failed.',
            status=AgentStatus.FAILURE,
        )

        mcp_client = agent.ollama_agent.mcp_client  # type: ignore[union-attr]
        mcp_connected = False

        # Setup phase: open the MCP connection (kept alive for the rest of
        # this method) and retrieve tools. Failures are logged but do not
        # prevent execution.
        try:
            self._log_info(f'AGENT [{agent_id}]: Starting execution pipeline...')
            if mcp_client is not None:
                await mcp_client.__aenter__()
                mcp_connected = True
                await mcp_client.ping()
                self._log_info(f'AGENT [{agent_id}]: MCP client connection verified')

            self._log_info(f'AGENT [{agent_id}]: Retrieving tools...')
            await agent.ollama_agent.retrieve_tools(  # type: ignore[union-attr]
                agent.lang_tools
            )
        except Exception as e:
            self._log_error(f'AGENT [{agent_id}]: Error during setup: {e}')

        # Execution phase: build graph and invoke the task
        try:
            if agent.graph is None:
                self._log_info(f'AGENT [{agent_id}]: Building graph...')
                await agent.make_graph()

            self._log_info(f'AGENT [{agent_id}]: Executing task...')
            result = await asyncio.wait_for(
                agent.graph.ainvoke(initial_state),  # type: ignore[attr-defined]
                timeout=self.agent_timeout,
            )
            execution_result.agent_result = result['messages'][-1]['content']

            final_status = agent.get_status()
            execution_result.status = final_status
            self._log_info(
                f'AGENT [{agent_id}]: Task completed with status: {final_status}'
            )
        except asyncio.CancelledError:
            self._log_info(
                f'AGENT [{agent_id}]: Cancelling execution by supervisor.'
            )
            raise
        except asyncio.TimeoutError:
            self._log_error(
                f'AGENT [{agent_id}]: Execution timed out after'
                f'{self.agent_timeout}s.'
            )
            agent.set_status(AgentStatus.FAILURE)
            execution_result.agent_result = (
                f'AGENT [{agent_id}]: Agent execution timed out after'
                f' {self.agent_timeout} seconds.'
            )
            execution_result.status = AgentStatus.FAILURE
        except Exception as e:
            self._log_error(f'AGENT [{agent_id}]: Error during execution: {e}')
            agent.set_status(AgentStatus.FAILURE)
            execution_result.status = AgentStatus.FAILURE
        finally:
            if mcp_connected:
                try:
                    await mcp_client.__aexit__(None, None, None)
                except Exception as e:
                    self._log_error(
                        f'AGENT [{agent_id}]: Error closing MCP client: {e}')

        # Store execution result in the registry
        self.registry.move_to_finished(agent_id, execution_result)

    def start(self) -> None:
        """
        Start the bounded pool of persistent worker threads.

        Spawns ``max_concurrent_agents`` daemon threads, each running
        ``_worker_loop``. Must be called once after initialization. Safe to
        call multiple times (subsequent calls are no-ops if workers are
        already running).
        """
        if self._worker_threads:
            self._log_info('Worker threads are already running.')
            return
        for i in range(self.max_concurrent_agents):
            thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f'agent-worker-{i}',
            )
            thread.start()
            self._worker_threads.append(thread)
        self._log_info(
            f'Started {self.max_concurrent_agents} agent worker thread(s).')

    def stop(self) -> None:
        """
        Signal every worker thread to stop and wait for termination.

        Enqueues one shutdown sentinel per worker so each of them observes
        it, exits its loop, and closes its event loop. An agent execution
        already in progress is allowed to finish; sentinels are only picked
        up once a worker is free.
        """
        for _ in self._worker_threads:
            self.registry.enqueue_shutdown_sentinel()
        for thread in self._worker_threads:
            thread.join(timeout=self.agent_timeout + 5.0)
        self._worker_threads = []
        self._log_info('Agent worker threads stopped.')

    def _worker_loop(self) -> None:
        """
        Run this worker's persistent loop, pulling tasks one at a time.

        Creates a single asyncio event loop for this thread and reuses it
        across every agent this worker processes in sequence, blocking with
        no polling delay between tasks. Exits when a shutdown sentinel
        (``None``) is received from the registry.
        """
        thread_name = threading.current_thread().name
        self._log_info(f'{thread_name}: worker started.')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                agent_task = self.registry.get_pending()
                if agent_task is None:
                    break
                self._run_task_on_loop(loop, agent_task)
        finally:
            loop.close()
            self._log_info(f'{thread_name}: worker exiting.')

    def _run_task_on_loop(self, loop: asyncio.AbstractEventLoop, agent_task) -> None:
        """
        Execute a single agent task to completion on the given event loop.

        Registers the agent as running in the registry, then blocks this
        worker thread until the coroutine completes or is cancelled.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            This worker's persistent event loop.
        agent_task : AgentTask
            The pending agent task containing the agent instance and initial state.

        """
        agent_id = agent_task.agent.get_id()
        thread_name = threading.current_thread().name
        self._log_info(
            f'AGENT [{agent_id}]: Dispatching on [{thread_name}] '
            f'(priority={agent_task.priority.name})'
        )

        coroutine_task = loop.create_task(
            self.run_agent(agent_task.agent, agent_task.input_state)
        )

        running_agent = RunningAgentsState(
            agent_id=agent_id,
            input_prompt=agent_task.input_state['messages'][0]['content'],
            coroutine_handler=coroutine_task,
            event_loop=loop,
        )
        self.registry.add_running(running_agent)

        try:
            loop.run_until_complete(coroutine_task)
        except asyncio.CancelledError:
            self._log_info(f'AGENT [{agent_id}]: Execution was cancelled.')
