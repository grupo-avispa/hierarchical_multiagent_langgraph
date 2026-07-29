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

"""Unit tests for AgentExecutor.run_agent, with a fake agent and MCP client."""

import asyncio

import pytest

from hierarchical_multiagent_langgraph.agent import AgentStatus
from hierarchical_multiagent_langgraph.agent_executor import AgentExecutor
from hierarchical_multiagent_langgraph.agent_registry import AgentRegistry


class FakeMcpClient:
    """
    Minimal stand-in for fastmcp's reentrant Client context manager.

    Mirrors fastmcp's nesting-counter behavior: only the outermost
    `__aenter__`/`__aexit__` pair actually connects/disconnects; nested
    `async with` calls just increment/decrement the counter and reuse the
    same session.
    """

    def __init__(self):
        self.nesting = 0
        self.connect_count = 0
        self.disconnect_count = 0

    async def __aenter__(self):
        if self.nesting == 0:
            self.connect_count += 1
        self.nesting += 1
        return self

    async def __aexit__(self, *exc_info):
        self.nesting -= 1
        if self.nesting == 0:
            self.disconnect_count += 1

    async def ping(self):
        pass


class FakeOllamaAgent:
    """Minimal stand-in for Ollama, exposing only what run_agent touches."""

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def retrieve_tools(self, lang_tools):
        pass


class FakeGraph:
    """Fake compiled LangGraph that simulates nested MCP tool calls."""

    def __init__(self, mcp_client, tool_call_count=2, raise_error=None):
        self.mcp_client = mcp_client
        self.tool_call_count = tool_call_count
        self.raise_error = raise_error

    async def ainvoke(self, initial_state):
        if self.raise_error is not None:
            raise self.raise_error
        # Simulate the reentrant `async with self.mcp_client:` calls made by
        # query_response()/Ollama.invoke() for each simulated tool call.
        if self.mcp_client is not None:
            for _ in range(self.tool_call_count):
                async with self.mcp_client:
                    pass
        return {'messages': [{'role': 'assistant', 'content': 'done'}]}


class FakeAgent:
    """Minimal stand-in for SinglePurposeAgent, exposing only what run_agent touches."""

    def __init__(self, agent_id, mcp_client, tool_call_count=2, raise_error=None):
        self._id = agent_id
        self.ollama_agent = FakeOllamaAgent(mcp_client)
        self.lang_tools = []
        self.graph = FakeGraph(mcp_client, tool_call_count, raise_error)
        self.status = AgentStatus.SUCCESS

    def get_id(self):
        return self._id

    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status

    async def make_graph(self):
        pass


def _initial_state():
    return {'messages': [{'role': 'user', 'content': 'do something'}]}


@pytest.mark.asyncio
async def test_run_agent_keeps_single_mcp_connection_open():
    """
    run_agent must open the MCP connection once and keep it open for the whole
    execution, instead of reconnecting for the ping, tool retrieval, and every
    tool call made inside the graph (see [B20]).
    """
    mcp_client = FakeMcpClient()
    agent = FakeAgent(agent_id=1, mcp_client=mcp_client, tool_call_count=3)
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0)

    await executor.run_agent(agent, _initial_state())

    assert mcp_client.connect_count == 1
    assert mcp_client.disconnect_count == 1
    assert mcp_client.nesting == 0


@pytest.mark.asyncio
async def test_run_agent_closes_mcp_connection_on_execution_error():
    """The MCP connection must still be closed if the graph execution raises."""
    mcp_client = FakeMcpClient()
    agent = FakeAgent(agent_id=2, mcp_client=mcp_client, raise_error=RuntimeError('boom'))
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0)

    await executor.run_agent(agent, _initial_state())

    assert mcp_client.connect_count == 1
    assert mcp_client.disconnect_count == 1
    summary = registry.get_summary()
    assert summary['finished'][0].status == AgentStatus.FAILURE


@pytest.mark.asyncio
async def test_run_agent_closes_mcp_connection_on_cancellation():
    """The MCP connection must still be closed if run_agent is cancelled mid-execution."""
    mcp_client = FakeMcpClient()

    class HangingGraph(FakeGraph):
        async def ainvoke(self, initial_state):
            async with self.mcp_client:
                await asyncio.sleep(10)

    agent = FakeAgent(agent_id=3, mcp_client=mcp_client)
    agent.graph = HangingGraph(mcp_client)
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0)

    task = asyncio.ensure_future(executor.run_agent(agent, _initial_state()))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert mcp_client.connect_count == 1
    assert mcp_client.disconnect_count == 1


@pytest.mark.asyncio
async def test_run_agent_without_mcp_client_still_completes():
    """run_agent must work normally for agents with no MCP client configured."""
    agent = FakeAgent(agent_id=4, mcp_client=None)
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=5.0)

    await executor.run_agent(agent, _initial_state())

    summary = registry.get_summary()
    assert summary['finished'][0].status == AgentStatus.SUCCESS
    assert summary['finished'][0].agent_result == 'done'


@pytest.mark.asyncio
async def test_run_agent_times_out_and_records_failure():
    """A hanging graph must be bounded by agent_timeout and recorded as FAILURE."""
    mcp_client = FakeMcpClient()

    class HangingGraph(FakeGraph):
        async def ainvoke(self, initial_state):
            await asyncio.sleep(10)

    agent = FakeAgent(agent_id=5, mcp_client=mcp_client)
    agent.graph = HangingGraph(mcp_client)
    registry = AgentRegistry()
    executor = AgentExecutor(registry=registry, agent_timeout=0.05)

    await executor.run_agent(agent, _initial_state())

    summary = registry.get_summary()
    assert summary['finished'][0].status == AgentStatus.FAILURE
    assert 'timed out' in summary['finished'][0].agent_result
