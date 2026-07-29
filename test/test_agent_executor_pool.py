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
Integration tests for AgentExecutor's bounded worker pool.

Exercises start()/enqueue()/stop() end to end (unlike test_agent_executor.py,
which calls run_agent() directly) to verify the pool actually bounds
concurrency and respects task priority under contention (see [B9]).
"""

import asyncio
import threading
import time

from hierarchical_multiagent_langgraph.agent import AgentStatus
from hierarchical_multiagent_langgraph.agent_executor import AgentExecutor
from hierarchical_multiagent_langgraph.agent_registry import AgentRegistry, AgentTask, TaskPriority


class ConcurrencyTracker:
    """Thread-safe recorder of how many fake graphs are executing at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current = 0
        self.max_seen = 0
        self.dispatch_order: list[str] = []

    def enter(self, name: str) -> None:
        with self._lock:
            self._current += 1
            self.max_seen = max(self.max_seen, self._current)
            self.dispatch_order.append(name)

    def exit(self, name: str) -> None:
        with self._lock:
            self._current -= 1


class RecordingGraph:
    """Fake compiled LangGraph that holds for a bit and records concurrency."""

    def __init__(self, name: str, tracker: ConcurrencyTracker, hold_time: float = 0.1):
        self.name = name
        self.tracker = tracker
        self.hold_time = hold_time

    async def ainvoke(self, initial_state):
        self.tracker.enter(self.name)
        try:
            await asyncio.sleep(self.hold_time)
        finally:
            self.tracker.exit(self.name)
        return {'messages': [{'role': 'assistant', 'content': 'done'}]}


class FakeOllamaAgent:
    """Minimal stand-in for Ollama: no MCP client, no-op tool retrieval."""

    def __init__(self):
        self.mcp_client = None

    async def retrieve_tools(self, lang_tools):
        pass


class FakeAgent:
    """Minimal stand-in for SinglePurposeAgent, exposing only what run_agent touches."""

    def __init__(self, agent_id, graph):
        self._id = agent_id
        self.ollama_agent = FakeOllamaAgent()
        self.lang_tools = []
        self.graph = graph
        self.status = AgentStatus.SUCCESS

    def get_id(self):
        return self._id

    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status

    async def make_graph(self):
        pass


def _initial_state(prompt: str):
    return {'messages': [{'role': 'user', 'content': prompt}]}


def _wait_until_finished(registry: AgentRegistry, expected_count: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(registry.get_summary()['finished']) >= expected_count:
            return
        time.sleep(0.02)
    raise AssertionError(
        f'Timed out waiting for {expected_count} finished agents '
        f"(got {len(registry.get_summary()['finished'])}).")


def test_executor_bounds_concurrent_agent_execution():
    """No more than max_concurrent_agents must run at the same time."""
    registry = AgentRegistry()
    tracker = ConcurrencyTracker()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0, max_concurrent_agents=2)
    executor.start()
    try:
        for i in range(6):
            agent = FakeAgent(i, RecordingGraph(f'agent-{i}', tracker, hold_time=0.1))
            registry.enqueue(AgentTask(
                agent=agent, input_state=_initial_state(f'task {i}'),
                priority=TaskPriority.MEDIUM))

        _wait_until_finished(registry, expected_count=6)
    finally:
        executor.stop()

    assert tracker.max_seen <= 2
    assert len(registry.get_summary()['finished']) == 6


def test_executor_dispatches_high_priority_ahead_of_queued_low_priority():
    """
    A high-priority agent enqueued while the single worker is busy must be
    picked up before an already-queued, lower-priority agent (see [B9]).
    """
    registry = AgentRegistry()
    tracker = ConcurrencyTracker()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0, max_concurrent_agents=1)
    executor.start()
    try:
        # Occupy the single worker so subsequent tasks queue up behind it.
        busy_agent = FakeAgent('busy', RecordingGraph('busy', tracker, hold_time=0.2))
        registry.enqueue(AgentTask(
            agent=busy_agent, input_state=_initial_state('busy'),
            priority=TaskPriority.MEDIUM))
        time.sleep(0.05)  # let the worker pick it up before queuing more

        low_agent = FakeAgent('low', RecordingGraph('low', tracker, hold_time=0.05))
        high_agent = FakeAgent('high', RecordingGraph('high', tracker, hold_time=0.05))
        registry.enqueue(AgentTask(
            agent=low_agent, input_state=_initial_state('low'), priority=TaskPriority.LOW))
        registry.enqueue(AgentTask(
            agent=high_agent, input_state=_initial_state('high'), priority=TaskPriority.HIGH))

        _wait_until_finished(registry, expected_count=3)
    finally:
        executor.stop()

    order_after_busy = [name for name in tracker.dispatch_order if name != 'busy']
    assert order_after_busy == ['high', 'low']


def test_executor_start_is_idempotent():
    """Calling start() while workers are already running must be a no-op."""
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0, max_concurrent_agents=3)
    executor.start()
    try:
        workers_after_first_start = list(executor._worker_threads)
        executor.start()
        assert executor._worker_threads == workers_after_first_start
    finally:
        executor.stop()


def test_executor_stop_joins_all_workers():
    """stop() must terminate every worker thread and leave the pool empty."""
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0, max_concurrent_agents=3)
    executor.start()
    workers = list(executor._worker_threads)

    executor.stop()

    assert executor._worker_threads == []
    assert all(not w.is_alive() for w in workers)
