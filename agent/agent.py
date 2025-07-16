from abc import ABC, abstractmethod
import logging
from typing import Union

from langchain_together import ChatTogether

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether]
    ):
        """initialize a new agent with LLM and tools"""
        pass