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
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
import itertools
import queue
from threading import Lock

from hierarchical_multiagent_langgraph.agent import AgentStatus, SinglePurposeAgent
from langgraph_base_ros.chat_template_render import Messages

# Priority used for the internal shutdown sentinel put on the pending queue.
# Lower than TaskPriority.HIGH so a shutdown request is always picked up by a
# free worker ahead of any real, still-pending task.
_SHUTDOWN_SENTINEL_PRIORITY = -1


class TaskPriority(IntEnum):
    """
    Priority levels for agent task scheduling.

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
        Mutex protecting concurrent access to the running/finished lists.
    _pending : queue.PriorityQueue
        Thread-safe priority queue of created but not-yet-started agents.
        Items are ``(priority, sequence, AgentTask | None)`` tuples; the
        sequence number breaks priority ties in FIFO order and keeps
        ``AgentTask`` (not orderable) out of tuple comparisons. A ``None``
        task is a shutdown sentinel (see ``enqueue_shutdown_sentinel``).
    _running : list[RunningAgentsState]
        Agents currently executing their tasks.
    _finished : deque[FinishedAgentsState]
        Completed agents with results. Bounded to ``max_finished`` entries;
        oldest results are evicted automatically once the limit is reached.
    _id_counter : int
        Auto-incrementing counter for unique agent IDs.

    """

    def __init__(self, max_finished: int = 20) -> None:
        """
        Initialize the registry with empty structures and the ID counter at 1.

        Parameters
        ----------
        max_finished : int
            Maximum number of finished agents retained in history. Oldest
            entries are evicted once the limit is exceeded. Defaults to 20.

        """
        self._lock = Lock()
        self._pending: queue.PriorityQueue = queue.PriorityQueue()
        self._pending_seq = itertools.count()
        self._running: list[RunningAgentsState] = []
        self._finished: deque[FinishedAgentsState] = deque(maxlen=max_finished)
        self._id_counter: int = 1

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
        Add a pending agent task to the priority queue.

        A worker blocked on ``get_pending()`` picks up the highest-priority
        task (lowest ``TaskPriority`` value) as soon as one is available,
        with no polling delay.

        Parameters
        ----------
        task : AgentTask
            The agent task to enqueue for later execution.

        """
        seq = next(self._pending_seq)
        self._pending.put((int(task.priority), seq, task))

    def get_pending(self, timeout: float | None = None) -> AgentTask | None:
        """
        Block until the next highest-priority task is available and return it.

        Parameters
        ----------
        timeout : float | None
            Maximum time to wait in seconds. None (the default, used by
            worker threads) blocks indefinitely with no polling delay.

        Returns
        -------
        AgentTask | None
            The next task to execute, in priority order. ``None`` means
            either a shutdown sentinel was received (the caller, a worker
            thread, should stop pulling further tasks) or ``timeout``
            expired with nothing pending.

        """
        try:
            _, _, task = self._pending.get(timeout=timeout)
        except queue.Empty:
            return None
        return task

    def enqueue_shutdown_sentinel(self) -> None:
        """
        Wake exactly one blocked worker so it can observe shutdown and exit.

        The sentinel uses a priority lower than any real ``TaskPriority``, so
        it is always picked up ahead of still-pending work. Called once per
        worker thread when stopping the executor.
        """
        seq = next(self._pending_seq)
        self._pending.put((_SHUTDOWN_SENTINEL_PRIORITY, seq, None))

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

    def cancel_running(self, agent_id: int, result: FinishedAgentsState) -> bool:
        """
        Atomically move a running agent to finished, guarding against races.

        Used by cancellation flows where the running agent may finish on
        its own (via ``move_to_finished``) during a slow network operation
        (e.g. stopping a behavior tree) that runs before this call. Checking
        membership and mutating the lists under a single lock prevents the
        agent from being recorded twice in ``_finished`` with contradictory
        statuses.

        Parameters
        ----------
        agent_id : int
            The unique identifier of the agent to cancel.
        result : FinishedAgentsState
            The finished agent state to record if the agent is still running.

        Returns
        -------
        bool
            True if the agent was running and has been moved to finished.
            False if the agent had already finished on its own, in which
            case no duplicate entry is added.

        """
        with self._lock:
            target = next(
                (a for a in self._running if a.agent_id == agent_id), None
            )
            if target is None:
                return False
            self._running.remove(target)
            self._finished.append(result)
            return True

    def get_agents_context(self, max_finished_in_context: int = 5) -> list[dict]:
        """
        Build a context list of all agents for system prompt rendering.

        Returns a snapshot of all running agents plus only the most recent
        finished ones, so the supervisor's system prompt does not grow
        unbounded as more tasks complete over the node's lifetime.

        Parameters
        ----------
        max_finished_in_context : int
            Maximum number of most-recent finished agents to include.
            Defaults to 5.

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
            recent_finished = list(self._finished)[-max_finished_in_context:]
            agents_list.extend([
                {
                    'id': agent.agent_id,
                    'query': agent.input_prompt,
                    'result': agent.agent_result,
                    'status': agent.status
                }
                for agent in recent_finished
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
            containing list copies. 'pending' is sorted by priority.

        """
        # queue.Queue guards its internal buffer with its own mutex, separate
        # from self._lock; snapshot it under that mutex rather than draining it.
        with self._pending.mutex:
            pending_items = sorted(self._pending.queue)
        pending = [task for _, _, task in pending_items if task is not None]

        with self._lock:
            return {
                'pending': pending,
                'running': list(self._running),
                'finished': list(self._finished),
            }
