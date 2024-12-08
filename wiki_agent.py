from typing_extensions import List
from langchain_cohere import ChatCohere
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
import requests


LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT=""
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT=""

os.environ["COHERE_API_KEY"] = ''



@tool
def search_wikipedia(query: str) -> str:
    """this function searches for the query in wikipedia and returns the title and snippet of the first search result"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        search_results = data['query']['search']
        
        if search_results:
            top_result = search_results[0]
            title = top_result['title']
            snippet = top_result['snippet']
            page_id = top_result['pageid']
            
            full_url = f"https://en.wikipedia.org/?curid={page_id}"
            return f"{title}: {snippet} Read more: {full_url}"
        else:
            return "No results found for '{}'".format(query)
    except Exception as e:
        return "An error occurred: {}".format(str(e))

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def search(query: str) -> str:
    """This function searches for the query in DuckDuckGo and returns the result"""
    return DuckDuckGoSearchRun().invoke(query)

class ChatBot:
    
    def __init__(self, tools: List):
        self.llm = ChatCohere(model="command-r-plus")
        self.llm_with_tools = self.llm.bind_tools(tools)
        

    def bind_tools(self) -> ChatCohere:
        return self.llm.bind_tools(self.tools)
    
    def run(self, query):
        messages = [HumanMessage(query)]
        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        print(ai_msg.content)

        for tool_call in ai_msg.tool_calls:
            selected_tool = {"add": add, "multiply": multiply, 'search_wikipedia': search_wikipedia}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            # print(tool_msg)
            messages.append(tool_msg)
            
        output = ''
        for message in messages:
            output += message.content + '\n'
        return self.llm_with_tools.invoke(messages).content

if __name__ == '__main__':
    query = "tell me about Pytorch"
    tools = [add, multiply, search_wikipedia]
    chatbot = ChatBot(tools)
    print(chatbot.run(query))
    
 
