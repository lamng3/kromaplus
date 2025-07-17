from abc import ABC, abstractmethod
import logging
from typing import Any, Awaitable, List, AsyncGenerator, Union

from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import BaseTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]]
    ):
        """initialize a new agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """synchronously invoke agent's main function"""
        pass

    @abstractmethod
    async def ainvoke(self, query: str, *args, **kwargs) -> Awaitable[Any]:
        """asynchronously invoke the agent's main function"""
        pass