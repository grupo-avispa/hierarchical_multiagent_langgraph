from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama
from enum import Enum


class AgentStatus(str, Enum):
    """Enumeration of possible agent statuses."""

    IDLE = 'idle'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class SinglePurposeAgent(LangGraphBase):
    """
    Represents a single-purpose agent with its configuration and status.

    Attributes:
        agent_id: Unique identifier for the agent.
        query: The task or query assigned to this agent.
        status: Current status of the agent (IDLE, RUNNING, SUCCESS, or FAILURE).
    """

    def __init__(
            self,
            logger=None,
            ollama_agent: Ollama | None = None,
            max_steps: int = 5
    ) -> None:
        """
        Initialize the Agent.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).
            ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
            max_steps (int): Maximum allowed steps before finishing interaction.

        Returns:
            None
        """
        super().__init__(
            logger=logger,
            ollama_agent=ollama_agent,
            max_steps=max_steps)
        if self.ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')
        self.ollama_agent: Ollama = self.ollama_agent

        self.id: int = -1  # Unique identifier for the agent
        self.status: AgentStatus = AgentStatus.IDLE  # Current status of the agent
        self.query: str = ''  # The task or query assigned to this agent

    def set_id(self, agent_id: int) -> None:
        """
        Set the unique identifier for the agent.

        Parameters:
            agent_id (int): Unique identifier to assign to the agent.

        Returns:
            None
        """
        self.id = agent_id

    def get_id(self) -> int:
        """
        Get the unique identifier of the agent.

        Returns:
            int: The unique identifier of the agent.
        """
        return self.id

    def get_status(self) -> str:
        """
        Get the current status of the agent.

        Returns:
            str: The current status of the agent.
        """
        return self.status

    def set_status(self, status: AgentStatus) -> None:
        """
        Set the current status of the agent.

        Parameters:
            status (AgentStatus): The new status to assign to the agent.

        Returns:
            None
        """
        self.status = status

    async def make_graph(self):
        """
        Build and compile the LangGraph workflow for this agent.

        This method should be overridden by subclasses to define
        the specific graph structure for the agent's task.

        Returns:
            None
        """
        pass
