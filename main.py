from agent import Agent
from dotenv import load_dotenv
import os
from langchain_together import ChatTogether

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key=together_api_key,
)

agent = Agent(llm)
print(agent.invoke("Tell me fun things to do in NYC"))