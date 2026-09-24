from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

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
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job openings for an AI engineer using langchain in bay area on linkedin and list their details")})
    print(result)
   
if __name__ == "__main__":
    main()