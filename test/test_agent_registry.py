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

"""Unit tests for AgentRegistry: pure in-memory logic, no ROS or LLM needed."""

import pytest

from hierarchical_multiagent_langgraph.agent_registry import (
    AgentRegistry,
    AgentTask,
    FinishedAgentsState,
    RunningAgentsState,
    TaskPriority,
)


def _make_task(priority: TaskPriority) -> AgentTask:
    """Build a minimal AgentTask; agent/input_state are unused by the registry itself."""
    return AgentTask(agent=None, input_state=None, priority=priority)


def test_next_id_increments_and_starts_at_one():
    """next_id() must be a thread-safe, auto-incrementing counter starting at 1."""
    registry = AgentRegistry()
    assert registry.next_id() == 1
    assert registry.next_id() == 2
    assert registry.next_id() == 3


def test_drain_all_pending_sorts_by_priority():
    """Pending tasks must be drained in ascending TaskPriority order (HIGH first)."""
    registry = AgentRegistry()
    registry.enqueue(_make_task(TaskPriority.LOW))
    registry.enqueue(_make_task(TaskPriority.HIGH))
    registry.enqueue(_make_task(TaskPriority.MEDIUM))

    drained = registry.drain_all_pending()

    assert [t.priority for t in drained] == [
        TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]
    # The queue must be empty after draining.
    assert registry.drain_all_pending() == []


def test_drain_all_pending_empty_queue_returns_empty_list():
    """Draining an empty registry must return an empty list, not None or raise."""
    registry = AgentRegistry()
    assert registry.drain_all_pending() == []


def test_pending_event_signals_and_clears():
    """enqueue() must signal wait_for_pending(); clear_pending_signal() must reset it."""
    registry = AgentRegistry()
    assert registry.wait_for_pending(timeout=0) is False

    registry.enqueue(_make_task(TaskPriority.MEDIUM))
    assert registry.wait_for_pending(timeout=0) is True

    registry.clear_pending_signal()
    assert registry.wait_for_pending(timeout=0) is False


def test_find_running_returns_none_when_absent():
    """find_running() must return None for an agent_id that isn't running."""
    registry = AgentRegistry()
    assert registry.find_running(agent_id=42) is None


def test_find_running_returns_matching_state():
    """find_running() must return the RunningAgentsState with the matching agent_id."""
    registry = AgentRegistry()
    state = RunningAgentsState(agent_id=7, input_prompt='do X')
    registry.add_running(state)

    found = registry.find_running(7)

    assert found is state


def test_move_to_finished_removes_from_running_and_appends_to_finished():
    """move_to_finished() must atomically move an agent from running to finished."""
    registry = AgentRegistry()
    registry.add_running(RunningAgentsState(agent_id=1, input_prompt='task'))

    result = FinishedAgentsState(agent_id=1, input_prompt='task', agent_result='ok')
    registry.move_to_finished(1, result)

    assert registry.find_running(1) is None
    context = registry.get_agents_context()
    assert context == [{'id': 1, 'query': 'task', 'result': 'ok', 'status': result.status}]


def test_cancel_running_moves_agent_and_returns_true():
    """cancel_running() must move a still-running agent to finished and return True."""
    registry = AgentRegistry()
    registry.add_running(RunningAgentsState(agent_id=5, input_prompt='task'))
    result = FinishedAgentsState(agent_id=5, input_prompt='task', agent_result='cancelled')

    was_cancelled = registry.cancel_running(5, result)

    assert was_cancelled is True
    assert registry.find_running(5) is None


def test_cancel_running_does_not_duplicate_already_finished_agent():
    """
    cancel_running() must not add a duplicate finished entry if the agent already
    finished on its own (e.g. via move_to_finished) before cancellation runs (see [B10]).
    """
    registry = AgentRegistry()
    registry.add_running(RunningAgentsState(agent_id=9, input_prompt='task'))

    # Simulate run_agent() completing on its own, concurrently with delete_agent's
    # network I/O, before cancel_running() gets a chance to run.
    real_result = FinishedAgentsState(agent_id=9, input_prompt='task', agent_result='done')
    registry.move_to_finished(9, real_result)

    cancel_result = FinishedAgentsState(
        agent_id=9, input_prompt='task', agent_result='cancelled')
    was_cancelled = registry.cancel_running(9, cancel_result)

    assert was_cancelled is False
    context = registry.get_agents_context()
    # Only the real (successful) result should be present -- no duplicate.
    assert len(context) == 1
    assert context[0]['result'] == 'done'


def test_finished_history_is_bounded_by_max_finished():
    """
    The finished-agents history must be bounded to max_finished entries;
    oldest results are evicted automatically (see [B5]).
    """
    registry = AgentRegistry(max_finished=3)
    for i in range(5):
        registry.add_running(RunningAgentsState(agent_id=i, input_prompt=f'task {i}'))
        registry.move_to_finished(
            i, FinishedAgentsState(agent_id=i, input_prompt=f'task {i}', agent_result='ok'))

    summary = registry.get_summary()
    assert len(summary['finished']) == 3
    # Only the three most recent (2, 3, 4) should remain; 0 and 1 were evicted.
    assert [f.agent_id for f in summary['finished']] == [2, 3, 4]


def test_get_agents_context_limits_finished_entries_shown_to_llm():
    """
    get_agents_context() must only include the most recent max_finished_in_context
    finished agents, independent of how many are retained in history (see [B5]).
    """
    registry = AgentRegistry(max_finished=20)
    for i in range(5):
        registry.add_running(RunningAgentsState(agent_id=i, input_prompt=f'task {i}'))
        registry.move_to_finished(
            i, FinishedAgentsState(agent_id=i, input_prompt=f'task {i}', agent_result='ok'))

    context = registry.get_agents_context(max_finished_in_context=2)

    finished_ids = [c['id'] for c in context if c['status'] != 'RUNNING']
    assert finished_ids == [3, 4]


def test_get_agents_context_includes_running_agents_in_full():
    """Running agents must always be included in full, regardless of the finished cap."""
    registry = AgentRegistry()
    registry.add_running(RunningAgentsState(agent_id=1, input_prompt='running task'))

    context = registry.get_agents_context(max_finished_in_context=0)

    assert context == [{'id': 1, 'query': 'running task', 'result': '', 'status': 'RUNNING'}]


def test_get_summary_returns_independent_copies():
    """get_summary() must return list copies that don't alias internal state."""
    registry = AgentRegistry()
    registry.enqueue(_make_task(TaskPriority.MEDIUM))

    summary = registry.get_summary()
    summary['pending'].clear()

    # Mutating the returned copy must not affect the registry's internal queue.
    assert len(registry.drain_all_pending()) == 1


@pytest.mark.parametrize('priority', [TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW])
def test_task_priority_values_order_correctly(priority):
    """TaskPriority must be an IntEnum where lower value means higher priority."""
    assert TaskPriority.HIGH < TaskPriority.MEDIUM < TaskPriority.LOW
    assert isinstance(priority.value, int)
