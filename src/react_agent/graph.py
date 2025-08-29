"""Define a custom Reasoning and Action agent."""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import render_text_description
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
load_dotenv()

from common.context import Context
from common.tools import create_tools
from common.utils import load_chat_model
from react_agent.state import InputState, State

# --- FIX: Simplify the function signature ---
async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Calls the LLM with the current state."""
    # --- FIX: Access context directly from the state ---
    context = state.context
    
    available_tools = await create_tools(context)
    tool_descriptions = render_text_description(available_tools)
    system_message = context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat(),
        tools=tool_descriptions,
    )
    model = load_chat_model(context.model).bind_tools(available_tools)
    messages = [{"role": "system", "content": system_message}, *state.messages]
    response = cast(AIMessage, await model.ainvoke(messages))

    if state.is_last_step and response.tool_calls:
        return {"messages": [AIMessage(id=response.id, content="Sorry, I could not find an answer in the allowed number of steps.")]}
    return {"messages": [response]}

# --- FIX: Simplify the function signature ---
async def dynamic_tools_node(state: State) -> Dict[str, List[ToolMessage]]:
    """Executes the tools that the LLM has decided to call."""
    # --- FIX: Access context directly from the state ---
    context = state.context
    
    available_tools = await create_tools(context)
    tool_node = ToolNode(available_tools)
    return cast(Dict, await tool_node.ainvoke(state))

# --- Graph Definition ---
builder = StateGraph(State, input_schema=InputState) # context_schema is no longer needed

# --- FIX: Add the plain functions directly as nodes ---
builder.add_node("call_model", call_model)
builder.add_node("tools", dynamic_tools_node)
# ----------------------------------------------------

builder.add_edge("__start__", "call_model")

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("Expected AIMessage at the end of the state.")
    if not last_message.tool_calls:
        return "__end__"
    return "tools"

builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

graph = builder.compile(name="ReAct Agent")