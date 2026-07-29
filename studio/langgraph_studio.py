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
This module sets up and initializes the LangGraphManager to create a LangGraph workflow.

It is meant to be used with LangGraph Studio.
Run it with the command: langgraph dev --allow-blocking
"""
from pathlib import Path

from hierarchical_multiagent_langgraph.supervisor import SupervisorManager
from langgraph_base_ros.ollama_utils import Ollama


async def make_graph():
    # Get the templates and params directories relative to this file
    package_dir = Path(__file__).parent.parent
    templates_path = package_dir / 'templates'
    params_path = package_dir / 'params'

    # Configuration for the SinglePurposeAgents created by the supervisor,
    # mirroring the defaults declared in main.py's _SPA_PARAM_DEFS.
    spa_params = {
        'mcp_servers_config': str(params_path / 'spa_mcp.json'),
        'system_prompt_file': str(templates_path / 'agent_system_prompt.jinja'),
        'template_type': 'qwen3',
        'template_file': str(templates_path / 'qwen3.jinja'),
        'model': 'qwen3:0.6b',
        'tool_call_pattern': '<tool_call>(.*?)</tool_call>',
        'available_tools': ['execute_behavior_tree'],
        'max_steps': 5,
        'think': False,
    }

    graph_manager = SupervisorManager(
        ollama_agent=Ollama(model='gpt-oss:20b'),
        system_prompt_path=str(templates_path / 'supervisor_system_prompt.jinja'),
        spa_params=spa_params,
    )
    await graph_manager.ollama_agent.retrieve_tools(graph_manager.supervisor_tools)
    await graph_manager.make_graph()
    return graph_manager.graph
