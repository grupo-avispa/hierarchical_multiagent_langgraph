"""
Thread-safe agent registry for managing agent lifecycle states.

This module provides the AgentRegistry class, which encapsulates all
agent lifecycle state management (pending, running, finished) behind
a thread-safe interface. It replaces direct manipulation of multiple
lists with a single registry that handles locking internally.

Key components:
    - AgentTask: Data structure for pending agent tasks.
    - RunningAgentsState: Tracks currently executing agents.
    - FinishedAgentsState: Tracks completed agent executions and results.
    - AgentRegistry: Thread-safe registry managing all agent states.
"""

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Event, Lock

from hierarchical_multiagent_langgraph.agent import AgentStatus, SinglePurposeAgent
from langgraph_base_ros.chat_template_render import Messages


class TaskPriority(IntEnum):
    """Priority levels for agent task scheduling.

    Lower integer values indicate higher priority. Agents with higher
    priority are dispatched first when multiple agents are pending.

    Attributes
    ----------
    HIGH : int
        Highest priority (0). Dispatched before all others.
    MEDIUM : int
        Default priority (1). Standard execution order.
    LOW : int
        Lowest priority (2). Dispatched after higher priority tasks.
    """

    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass
class AgentTask:
    """
    Encapsulate a pending agent task awaiting execution.

    Represents a task created by the supervisor's LLM but not yet started.
    Tasks are stored in a queue and executed asynchronously by the supervisor
    when resources become available.

    Attributes
    ----------
    agent : SinglePurposeAgent | None
        The SinglePurposeAgent instance that will execute the task.
    input_state : Messages | None
        The initial message state containing the task description and context.
    """

    agent: SinglePurposeAgent = None  # type: ignore[assignment]
    input_state: Messages = None  # type: ignore[assignment]
    priority: TaskPriority = field(default=TaskPriority.MEDIUM)


@dataclass
class RunningAgentsState:
    """
    Track the execution state of a currently running agent.

    Maintain metadata about an agent that is actively executing its task.
    Enables monitoring, timeout management, and graceful cancellation of
    long-running agents.

    Attributes
    ----------
    agent_id : int
        Unique identifier of the running agent.
    input_prompt : str
        The original input prompt/task description given to the agent.
    coroutine_handler : asyncio.Task | None
        The asyncio Task object managing the agent's concurrent execution.
    event_loop : asyncio.AbstractEventLoop | None
        Reference to the event loop running the agent's coroutine.
    """

    agent_id: int = -1
    input_prompt: str = ''
    coroutine_handler: asyncio.Task = None  # type: ignore[assignment]
    event_loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]


@dataclass
class FinishedAgentsState:
    """
    Capture the final state and results of a completed agent task.

    Contains the execution summary for an agent that has finished its task,
    whether successfully or with failure. Aggregated by the supervisor to
    synthesize final responses.

    Attributes
    ----------
    agent_id : int
        Unique identifier of the finished agent.
    input_prompt : str
        The original task description given to the agent.
    agent_result : str
        The result or output produced by the agent's execution.
    status : AgentStatus
        The final execution status (SUCCESS, FAILURE, or IDLE).
    """

    agent_id: int = -1
    input_prompt: str = ''
    agent_result: str = ''
    status: AgentStatus = AgentStatus.IDLE


class AgentRegistry:
    """
    Thread-safe registry for managing agent lifecycle states.

    Encapsulates pending, running, and finished agent lists behind a
    unified interface with internal locking. All public methods are
    thread-safe and can be called from multiple threads concurrently.

    Attributes
    ----------
    _lock : Lock
        Mutex protecting concurrent access to agent lists.
    _pending : list[AgentTask]
        Queue of created but not-yet-started agents.
    _running : list[RunningAgentsState]
        Agents currently executing their tasks.
    _finished : list[FinishedAgentsState]
        Completed agents with results.
    _id_counter : int
        Auto-incrementing counter for unique agent IDs.
    """

    def __init__(self) -> None:
        """Initialize the registry with empty lists, counter at 1, and pending event."""
        self._lock = Lock()
        self._pending: list[AgentTask] = []
        self._running: list[RunningAgentsState] = []
        self._finished: list[FinishedAgentsState] = []
        self._id_counter: int = 1
        # Event signaled when new pending agents are available
        self._pending_event = Event()

    def next_id(self) -> int:
        """
        Generate the next unique agent ID.

        Thread-safe auto-increment of the internal counter.

        Returns
        -------
        int
            The next unique agent identifier.
        """
        with self._lock:
            agent_id = self._id_counter
            self._id_counter += 1
        return agent_id

    def enqueue(self, task: AgentTask) -> None:
        """
        Add a pending agent task to the queue and signal the consumer.

        Appends the task under the lock and then sets the pending event
        to wake any waiting consumer thread.

        Parameters
        ----------
        task : AgentTask
            The agent task to enqueue for later execution.
        """
        with self._lock:
            self._pending.append(task)
        # Signal the consumer thread that work is available
        self._pending_event.set()

    def pop_pending(self) -> AgentTask | None:
        """
        Pop the next pending agent task from the queue.

        Returns
        -------
        AgentTask | None
            The next pending task, or None if the queue is empty.
        """
        with self._lock:
            if self._pending:
                return self._pending.pop(0)
        return None

    def wait_for_pending(self, timeout: float | None = None) -> bool:
        """
        Block until a pending agent is available or timeout expires.

        Parameters
        ----------
        timeout : float | None
            Maximum time to wait in seconds. None waits indefinitely.

        Returns
        -------
        bool
            True if the event was set (pending agent available),
            False if the timeout expired.
        """
        return self._pending_event.wait(timeout=timeout)

    def clear_pending_signal(self) -> None:
        """Clear the pending signal after consuming pending agents."""
        self._pending_event.clear()

    def wake_pending(self) -> None:
        """
        Force-wake any thread waiting for pending agents.

        Used during shutdown to unblock the consumer thread so it can
        check the shutdown flag and exit cleanly.
        """
        self._pending_event.set()

    def drain_all_pending(self) -> list[AgentTask]:
        """
        Drain all pending tasks sorted by priority (highest first).

        Atomically removes all pending tasks from the queue and returns
        them sorted by ascending ``TaskPriority`` value (lower value =
        higher priority).

        Returns
        -------
        list[AgentTask]
            All previously pending tasks sorted by priority, or an
            empty list if the queue was empty.
        """
        with self._lock:
            tasks = list(self._pending)
            self._pending.clear()
        tasks.sort(key=lambda t: t.priority)
        return tasks

    def add_running(self, state: RunningAgentsState) -> None:
        """
        Register an agent as currently running.

        Parameters
        ----------
        state : RunningAgentsState
            The running agent metadata to track.
        """
        with self._lock:
            self._running.append(state)

    def find_running(self, agent_id: int) -> RunningAgentsState | None:
        """
        Find a running agent by its ID without removing it.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent to find.

        Returns
        -------
        RunningAgentsState | None
            The running agent state, or None if not found.
        """
        with self._lock:
            return next(
                (a for a in self._running if a.agent_id == agent_id),
                None
            )

    def remove_running(self, agent_id: int) -> None:
        """
        Remove a running agent by its ID.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent to remove.
        """
        with self._lock:
            self._running = [
                a for a in self._running if a.agent_id != agent_id
            ]

    def move_to_finished(self, agent_id: int, result: FinishedAgentsState) -> None:
        """
        Move an agent from running to finished state atomically.

        Removes the agent from the running list and appends the result
        to the finished list in a single locked transaction.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent that finished.
        result : FinishedAgentsState
            The finished agent state with execution results.
        """
        with self._lock:
            self._running = [
                a for a in self._running if a.agent_id != agent_id
            ]
            self._finished.append(result)

    def add_finished(self, result: FinishedAgentsState) -> None:
        """
        Add a finished agent result directly.

        Used when an agent is cancelled and needs to be marked as finished
        without going through the normal running→finished transition.

        Parameters
        ----------
        result : FinishedAgentsState
            The finished agent state with execution results.
        """
        with self._lock:
            self._finished.append(result)

    def get_agents_context(self) -> list[dict]:
        """
        Build a context list of all agents for system prompt rendering.

        Returns a snapshot of running and finished agents as dictionaries
        suitable for Jinja2 template rendering.

        Returns
        -------
        list[dict]
            List of agent context dictionaries with keys: id, query,
            result, status.
        """
        with self._lock:
            agents_list = [
                {
                    'id': agent.agent_id,
                    'query': agent.input_prompt,
                    'result': '',
                    'status': 'RUNNING'
                }
                for agent in self._running
            ]
            agents_list.extend([
                {
                    'id': agent.agent_id,
                    'query': agent.input_prompt,
                    'result': agent.agent_result,
                    'status': agent.status
                }
                for agent in self._finished
            ])
        return agents_list

    def get_summary(self) -> dict[str, list]:
        """
        Get a snapshot of all agent lists for logging purposes.

        Returns a dictionary with copies of pending, running, and finished
        agent lists for safe iteration outside the lock.

        Returns
        -------
        dict[str, list]
            Dictionary with keys 'pending', 'running', 'finished'
            containing list copies.
        """
        with self._lock:
            return {
                'pending': list(self._pending),
                'running': list(self._running),
                'finished': list(self._finished),
            }
