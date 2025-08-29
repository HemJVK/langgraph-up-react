"""Tests for the ReAct agent graph execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage, HumanMessage

from common.context import Context
from react_agent.graph import graph

from dotenv import load_dotenv
load_dotenv()

pytestmark = pytest.mark.asyncio

@patch("react_agent.graph.load_chat_model")
async def test_react_agent_simple_passthrough_mocked(
    mock_load_chat_model: MagicMock,
) -> None:
    """Tests the agent's ability to answer a simple question directly."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="The founder is Harrison Chase.")
    mock_model.bind_tools.return_value = mock_model
    mock_load_chat_model.return_value = mock_model

    # --- FIX: Pass context as part of the main input dictionary ---
    graph_input = {
        "messages": [("user", "Who is the founder of LangChain?")],
        "context": Context() 
    }
    res = await graph.ainvoke(graph_input)
    # -----------------------------------------------------------

    assert len(res["messages"]) == 2
    final_response = res["messages"][-1].content
    assert "Harrison Chase" in final_response
    mock_model.ainvoke.assert_awaited_once()

@patch("react_agent.graph.create_tools")
@patch("react_agent.graph.load_chat_model")
async def test_react_agent_full_tool_cycle_mocked(
    mock_load_chat_model: MagicMock, mock_create_tools: MagicMock
) -> None:
    """Tests the full ReAct (Reason-Act) loop of the agent."""
    mock_search_tool = AsyncMock()
    mock_search_tool.name = "tavily_search_results_json"
    mock_search_tool.invoke.return_value = "LangChain was founded by Harrison Chase in 2022."
    mock_create_tools.return_value = [mock_search_tool]

    tool_call = ToolCall(name="tavily_search_results_json", args={"query": "founder of LangChain"}, id="123")
    first_model_response = AIMessage(content="", tool_calls=[tool_call])
    second_model_response = AIMessage(content="Harrison Chase is the founder of LangChain.")

    mock_model = AsyncMock()
    mock_model.ainvoke.side_effect = [first_model_response, second_model_response]
    mock_model.bind_tools.return_value = mock_model
    mock_load_chat_model.return_value = mock_model

    # --- FIX: Pass context as part of the main input dictionary ---
    graph_input = {
        "messages": [("user", "Who is the founder of LangChain?")],
        "context": Context()
    }
    res = await graph.ainvoke(graph_input)
    # -----------------------------------------------------------

    assert len(res["messages"]) == 4
    assert "Harrison Chase" in res["messages"][3].content
    mock_create_tools.assert_awaited()
    assert mock_model.ainvoke.await_count == 2