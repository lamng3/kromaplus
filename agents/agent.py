from abc import ABC, abstractmethod
import json
from pathlib import Path
import logging
from typing import Any, Awaitable, List, AsyncGenerator, Union, Sequence

from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.tool import ToolMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]]
    ):
        """Initialize a new agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Synchronously invoke agent's main function"""
        pass


class Agent(AgentMeta):
    """AI agent with tool-calling capabilities."""

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        tools_path: Path = Path("templates/example_tools.json"),
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        skills: list[str] = ["You can answer the user question with tools"],
    ):
        """
        Initialize agent with a language model, a set of tools, a description, and a set of skills.
        """
        self.llm = llm
        self.tools = tools
        self.description = description
        self.skills = skills

        # initialize tools
        self.tools_path = None
        if tools_path:
            self.tools_path = (
                Path(tools_path) if isinstance(tools_path, str) else tools_path
            )
        else:
            self.tools_path = Path("templates/example_tools.json")
        if self.tools_path and (self.tools_path.suffix != ".json"):
            raise ValueError(
                "tools_path must be json format ending with .json. For example, 'templates/tools.json'"
            )
        self.tools_path.parent.mkdir(parents=True, exist_ok=True)
        
    def prompt_template(self, query: str, *arg, **kwargs) -> str:
        try:
            tools = json.loads(self.tools_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        # config this prompt to better fit with task
        prompt = (
            "You are given an ontology developing task and a list of available tools.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(tools)}\n"
            "------------------------\n"
            "Instructions:\n"
            "- Let's answer in a natural, clear, and detailed way.\n"
            "- You need to think about whether the question need to use Tools?\n"
            "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"tool_type": "Type of tool. Only get one of three values ["function", "module", "mcp"]"\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "- Let's say I don't know and suggest where to search if you are unsure the answer.\n"
            "- Not make up anything.\n"
        )
        return prompt

    def invoke(self, query: Union[str, Sequence[BaseMessage]], **kwargs) -> Any:
        """Select and execute a took based on the task description"""
        prompt = self.prompt_template(query=query)
        skills = "- " + "- ".join(self.skills)
        messages: Sequence[BaseMessage] = [
            SystemMessage(
                content=f"{self.description}\nHere is your skills:\n{skills}"
            ),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        if hasattr(response, "content"):
            result = response.content
        else:
            result = str(response)

        logger.info(f"Agent response: {result}")
        return result