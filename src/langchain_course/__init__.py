from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the Agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answers and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

load_dotenv()


# tavily = TavilyClient()

# @tool
# def search(query:str)->str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for 
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job openings for an AI engineer using langchain in bay area on linkedin and list their details")})
    print(result)
   
if __name__ == "__main__":
    main()