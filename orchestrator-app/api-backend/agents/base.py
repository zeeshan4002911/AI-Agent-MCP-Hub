import os
from typing import Annotated, TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
if (
    not os.environ["HUGGINGFACEHUB_API_TOKEN"]
    or os.environ["HUGGINGFACEHUB_API_TOKEN"] == ""
):
    raise Exception("HUGGINGFACEHUB_API_TOKEN is required as env variable")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
    temperature=0.7,
)

llm = ChatHuggingFace(llm=llm_endpoint)


class AgentState(TypedDict):
    topic: str
    draft: str
    review: str


def researcher_node(state: AgentState):
    print("Running Researcher node")
    prompt = f"Write a brief, one-paragraph technical summary about: {state['topic']}. Focus on facts."
    response = llm.invoke(prompt)

    return {"draft": response}


def editor_node(state: AgentState):
    print("Running Editor node")
    prompt = f"Polsh this text to make it sound highly professional and corporate: {state['draft']}"
    response = llm.invoke(prompt)

    return {"review": response}


def sample_graph():
    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("editor", editor_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "editor")
    builder.add_edge("editor", END)

    graph = builder.compile()

    initial_input = {"topic": "The importance of fine-tuning LLMs with small datasets"}
    final_state = graph.invoke(initial_input)

    print("Graph Output")
    print(final_state["review"])
    return final_state["review"]
