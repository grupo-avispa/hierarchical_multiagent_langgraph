"""
This module sets up and initializes the LangGraphManager to create a LangGraph workflow.

It is meant to be used with LangGraph Studio.
Run it with the command: langgraph dev --allow-blocking
"""
from pathlib import Path
from hierarchical_multiagent_langgraph.supervisor import SupervisorManager
from langgraph_base_ros.ollama_utils import Ollama


async def make_graph():
    # Get the templates directory path relative to this file
    current_dir = Path(__file__).parent
    templates_path = str(current_dir.parent / 'templates')

    with open(templates_path + '/system_prompt.jinja', 'r') as f:
        system_prompt = f.read()

    graph_manager = SupervisorManager(
        ollama_agent=Ollama(model='gpt-oss:20b'),
        system_prompt=system_prompt)
    await graph_manager.make_graph()
    return graph_manager.graph
