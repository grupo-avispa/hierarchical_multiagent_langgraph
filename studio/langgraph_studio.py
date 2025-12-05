"""
This module sets up and initializes the LangGraphManager to create a LangGraph workflow.

It is meant to be used with LangGraph Studio.
Run it with the command: langgraph dev --allow-blocking
"""
from hierarchical_multiagent_langgraph.supervisor import SupervisorManager
from langgraph_base_ros.ollama_utils import Ollama


async def make_graph():
    graph_manager = SupervisorManager(ollama_agent=Ollama())
    await graph_manager.make_graph()
    return graph_manager.graph
