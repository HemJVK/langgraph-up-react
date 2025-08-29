"""
End-to-end tests for the LangGraph ReAct agent API.

These tests require the LangGraph server to be running.
(e.g., via `make dev`)
"""

import pytest

from tests.test_data import TestQuestions

# Mark all tests in this file as e2e
pytestmark = pytest.mark.e2e

from dotenv import load_dotenv

load_dotenv()


async def test_api_simple_knowledge_question(
    langgraph_client, assistant_id, helpers
) -> None:
    """Test the agent's ability to answer a direct question without using tools."""
    # Arrange
    thread = await langgraph_client.threads.create()
    question_data = TestQuestions.SIMPLE_KNOWLEDGE
    run_input = {"messages": [{"role": "human", "content": question_data["question"]}]}

    # Act
    result = await langgraph_client.runs.wait(
        thread_id=thread["thread_id"], assistant_id=assistant_id, input=run_input
    )

    # Assert
    messages = result["messages"]
    helpers.assert_valid_response(messages, expected_content=question_data["expected_in_response"])


async def test_api_uses_web_search_tool(
    langgraph_client, assistant_id, helpers
) -> None:
    """Test that the agent correctly uses the web search tool for a timely question."""
    # Arrange
    thread = await langgraph_client.threads.create()
    question_data = TestQuestions.TOOL_USAGE_KNOWLEDGE
    run_input = {"messages": [{"role": "human", "content": question_data["question"]}]}

    # Act
    result = await langgraph_client.runs.wait(
        thread_id=thread["thread_id"], assistant_id=assistant_id, input=run_input
    )

    # Assert
    messages = result["messages"]
    helpers.assert_valid_response(messages)
    # Verify that the correct tool was used to find the answer
    helpers.assert_tool_usage(messages, tool_name=question_data["tool_to_use"])


async def test_api_uses_python_repl_tool(
    langgraph_client, assistant_id, helpers
) -> None:
    """Test that the agent correctly uses the Python REPL for a calculation."""
    # Arrange
    thread = await langgraph_client.threads.create()
    question_data = TestQuestions.TOOL_USAGE_CALCULATION
    run_input = {"messages": [{"role": "human", "content": question_data["question"]}]}

    # Act
    result = await langgraph_client.runs.wait(
        thread_id=thread["thread_id"], assistant_id=assistant_id, input=run_input
    )

    # Assert
    messages = result["messages"]
    helpers.assert_valid_response(messages, expected_content=question_data["expected_in_response"])
    # Verify that the Python tool was used for the calculation
    helpers.assert_tool_usage(messages, tool_name=question_data["tool_to_use"])


async def test_api_maintains_conversation_context(
    langgraph_client, assistant_id, helpers
) -> None:
    """Test that the agent can remember information across multiple turns in a thread."""
    # Arrange
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    # First turn: provide a piece of information
    input1 = {"messages": [{"role": "human", "content": "My name is Bob."}]}
    await langgraph_client.runs.wait(thread_id=thread_id, assistant_id=assistant_id, input=input1)

    # Second turn: ask a question that relies on the previous turn's context
    input2 = {"messages": [{"role": "human", "content": "What is my name?"}]}

    # Act
    result2 = await langgraph_client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input2
    )

    # Assert
    messages = result2["messages"]
    helpers.assert_valid_response(messages, expected_content="Bob")