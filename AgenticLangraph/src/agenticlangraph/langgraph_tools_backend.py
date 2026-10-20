import requests
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import sqlite3


load_dotenv()

model = ChatOllama(
    model= "llama3.2:latest"
)

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operator: str) -> dict:
    """ 
    perform basic arthematic operation on two numbers.
    Supported operations : add, sub, mul, div
    """
    try:
        if operator == "add":
            result = first_num + second_num
        elif operator == "sub":
            result = first_num - second_num
        elif operator == "mul":
            result = first_num * second_num
        elif operator == "div":
            if second_num ==0:
                return {"error": "Division by zero is not allowed"}
            result = first_num/second_num

        else:
            return {"error": f"Unsuported operator {operator}"}
        return {"first_num":first_num, "second_num": second_num, "operator": operator, "result" : result}
    except Exception as e:
        return {"error" : str(e)}

#Make tool list
tools = [search_tool, calculator]

# Making LLM tool-aware
llm_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer= checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)