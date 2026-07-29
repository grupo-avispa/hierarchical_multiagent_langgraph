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

"""Unit tests for SupervisorManager.delete_agent's event-loop selection."""

import asyncio
from unittest.mock import MagicMock

import pytest

from hierarchical_multiagent_langgraph.agent_registry import RunningAgentsState
from hierarchical_multiagent_langgraph.supervisor import SupervisorManager


@pytest.fixture
def supervisor_manager(tmp_path):
    """Build a SupervisorManager with a mocked Ollama agent and MCP client."""
    prompt_path = tmp_path / 'system_prompt.jinja'
    prompt_path.write_text('You are a supervisor.')

    mock_ollama = MagicMock()
    mock_ollama.mcp_client = MagicMock()

    manager = SupervisorManager(
        ollama_agent=mock_ollama,
        spa_params={},
        system_prompt_path=str(prompt_path),
    )
    return manager


def _get_delete_agent_func(manager):
    """Extract the raw delete_agent callable from the wrapped LangChain tool."""
    tool_entry = next(t for t in manager.supervisor_tools if t['name'] == 'delete_agent')
    return tool_entry['tool_object'].func


def test_delete_agent_uses_node_loop_not_agent_loop(supervisor_manager, monkeypatch):
    """
    delete_agent must schedule the stop_behavior_tree MCP call on the
    supervisor's node_loop, which owns mcp_client's connection, rather than on
    the agent's own event loop, which does not (see [B3]).
    """
    node_loop = asyncio.new_event_loop()
    agent_loop = asyncio.new_event_loop()
    supervisor_manager.node_loop = node_loop

    async def fake_call_tool(*args, **kwargs):
        return 'ok'

    supervisor_manager.ollama_agent.mcp_client.call_tool = (
        lambda *a, **kw: fake_call_tool(*a, **kw))

    recorded_loops = []

    class FakeFuture:
        def result(self, timeout=None):
            return 'stopped'

    def fake_run_coroutine_threadsafe(coro, loop):
        recorded_loops.append(loop)
        coro.close()
        return FakeFuture()

    monkeypatch.setattr(
        'hierarchical_multiagent_langgraph.supervisor.asyncio.run_coroutine_threadsafe',
        fake_run_coroutine_threadsafe,
    )

    supervisor_manager.registry.add_running(
        RunningAgentsState(
            agent_id=1,
            input_prompt='do X',
            coroutine_handler=MagicMock(),
            event_loop=agent_loop,
        )
    )

    delete_agent = _get_delete_agent_func(supervisor_manager)
    delete_agent(agent_id=1)

    assert recorded_loops == [node_loop]
    assert supervisor_manager.registry.find_running(1) is None


def test_delete_agent_skips_mcp_call_when_node_loop_not_configured(
        supervisor_manager, monkeypatch):
    """If node_loop was never injected, delete_agent must not guess a loop to use."""
    supervisor_manager.node_loop = None
    agent_loop = asyncio.new_event_loop()

    called = []
    monkeypatch.setattr(
        'hierarchical_multiagent_langgraph.supervisor.asyncio.run_coroutine_threadsafe',
        lambda *a, **kw: called.append((a, kw)),
    )

    supervisor_manager.registry.add_running(
        RunningAgentsState(
            agent_id=2,
            input_prompt='do Y',
            coroutine_handler=MagicMock(),
            event_loop=agent_loop,
        )
    )

    delete_agent = _get_delete_agent_func(supervisor_manager)
    delete_agent(agent_id=2)

    assert called == []
    assert supervisor_manager.registry.find_running(2) is None
