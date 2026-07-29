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
graph building, task execution, and result collection. Each agent runs in
its own asyncio event loop within a dedicated thread.

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
    Manage async execution of SinglePurposeAgent instances in dedicated threads.

    Uses an event-driven consumer pattern: a background thread waits on the
    registry's pending event and spawns a new thread for each pending agent
    when signaled. Replaces the previous polling-based timer approach for
    lower latency and reduced resource usage.

    Attributes
    ----------
    registry : AgentRegistry
        Thread-safe registry for agent lifecycle state management.
    agent_timeout : float
        Maximum time in seconds for a single agent execution.
    logger : logging.Logger | None
        Optional logger instance for debug/info/warning output.

    """

    def __init__(
        self,
        registry: AgentRegistry,
        agent_timeout: float = 120.0,
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
        logger : logging.Logger | None
            Optional logger for debug/info/warning output. If None,
            uses Python's standard logging module. Defaults to None.

        """
        self.registry = registry
        self.agent_timeout = agent_timeout
        self.logger = logger
        # Shutdown flag for the consumer thread
        self._shutdown_event = threading.Event()
        # Background consumer thread reference
        self._consumer_thread: threading.Thread | None = None

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

    def consume_pending_agents(self) -> None:
        """
        Drain all pending agents, sort by priority, and execute each in a thread.

        Atomically drains all pending agents from the registry, sorts them
        by priority (highest first), and spawns a dedicated daemon thread
        for each one. Each thread creates its own asyncio event loop and
        blocks until agent execution completes.

        This method is non-blocking from the caller's perspective: it returns
        immediately after spawning the threads.
        """
        pending_tasks = self.registry.drain_all_pending()
        if not pending_tasks:
            return

        self._log_info(
            f'Dispatching {len(pending_tasks)} pending agent(s) '
            f'sorted by priority...'
        )

        for agent_task in pending_tasks:
            agent_id = agent_task.agent.get_id()
            self._log_info(
                f'AGENT [{agent_id}]: Spawning daemon thread for agent execution '
                f'(priority={agent_task.priority.name})'
            )
            thread = threading.Thread(
                target=self._execute_agent_task,
                args=(agent_task,),
                daemon=True,
                name=f'agent-{agent_id}',
            )
            thread.start()

    def start(self) -> None:
        """
        Start the background consumer thread for event-driven agent execution.

        Spawns a daemon thread that waits on the registry's pending event.
        When new agents are enqueued, the thread wakes up, consumes all
        pending agents, and spawns a dedicated thread for each one.

        Must be called once after initialization. Safe to call multiple times
        (subsequent calls are no-ops if the thread is already running).
        """
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._log_info('Consumer thread is already running.')
            return
        self._shutdown_event.clear()
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name='agent-consumer'
        )
        self._consumer_thread.start()
        self._log_info('Agent consumer thread started.')

    def stop(self) -> None:
        """
        Signal the consumer thread to stop and wait for termination.

        Sets the internal shutdown flag and wakes the consumer thread via
        the registry's pending event so it can check the flag and exit
        its loop. Running agent threads are daemon threads and will be
        terminated when the process exits.
        """
        self._shutdown_event.set()
        # Wake the consumer thread if it is blocked waiting for pending agents
        self.registry.wake_pending()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=5.0)
            self._consumer_thread = None
        self._log_info('Agent consumer thread stopped.')

    def _consumer_loop(self) -> None:
        """
        Wait for pending agents and execute them in dedicated threads.

        Blocks on the registry's pending event. When agents are enqueued,
        wakes up, clears the signal, consumes all pending agents (each in
        its own thread), and resumes waiting. Exits when the shutdown event
        is set.
        """
        self._log_info('Consumer loop started, waiting for pending agents...')
        while not self._shutdown_event.is_set():
            # Block until pending agents are available or shutdown is signaled
            self.registry.wait_for_pending()
            if self._shutdown_event.is_set():
                break
            # Clear the signal and drain all pending agents
            self.registry.clear_pending_signal()
            self.consume_pending_agents()
        self._log_info('Consumer loop exiting.')

    def _execute_agent_task(self, agent_task) -> None:
        """
        Execute a single agent task in a dedicated thread with its own event loop.

        Creates a new asyncio event loop, schedules the agent execution coroutine,
        registers the agent as running in the registry, and blocks until the
        coroutine completes or is cancelled.

        Parameters
        ----------
        agent_task : AgentTask
            The pending agent task containing the agent instance and initial state.

        """
        agent_id = agent_task.agent.get_id()
        self._log_info(
            f'AGENT [{agent_id}]: Starting execution in thread '
            f'[{threading.current_thread().name}]'
        )

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Schedule the agent execution coroutine
        coroutine_task = loop.create_task(
            self.run_agent(agent_task.agent, agent_task.input_state)
        )

        # Register as running in the registry
        running_agent = RunningAgentsState(
            agent_id=agent_id,
            input_prompt=agent_task.input_state['messages'][0]['content'],
            coroutine_handler=coroutine_task,
            event_loop=loop,
        )
        self.registry.add_running(running_agent)

        # Block this thread until the agent task completes
        try:
            self._log_info(
                f'AGENT [{agent_id}]: Working on agent in thread '
                f'[{threading.current_thread().name}]...'
            )
            loop.run_until_complete(coroutine_task)
        except asyncio.CancelledError:
            self._log_info(f'AGENT [{agent_id}]: Execution was cancelled.')
        finally:
            loop.close()
