"""
Tests the agent graph's integration with various model providers.

This file uses mocking and parametrization to ensure that the graph correctly
initializes and invokes models from different supported providers (OpenAI, Groq,
Ollama, Gemini) without making actual network calls.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from common.context import Context
from react_agent.graph import graph

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# A list of model providers to test.
# When a new provider is added to models.py, just add its info here.
SUPPORTED_PROVIDERS = [
    ("openai", "gpt-4o-mini", "common.models.ChatOpenAI"),
    ("groq", "llama3-70b-8192", "common.models.ChatGroq"),
    ("ollama", "llama3", "common.models.ChatOllama"),
    ("gemini", "gemini-1.5-pro", "common.models.ChatGoogleGenerativeAI"),
]


@pytest.mark.parametrize("provider, model_name, mock_path", SUPPORTED_PROVIDERS)
async def test_agent_workflow_with_all_providers(
    provider: str, model_name: str, mock_path: str
) -> None:
    """
    Tests the full agent workflow for a given model provider using mocks.

    This parametrized test verifies that for each supported provider, the
    agent correctly:
    1. Initializes the appropriate model class.
    2. Invokes the model with the correct messages.
    3. Returns the mocked response, completing the graph run.
    """
    # --- Arrange ---

    # Use patch to mock the specific ChatModel class for the provider being tested.
    with patch(mock_path) as mock_chat_class:
        # 1. Mock the model instance and its methods
        mock_model_instance = MagicMock()
        mock_model_instance.bind_tools.return_value = mock_model_instance
        
        # The model will return a simple, direct answer
        mock_response = AIMessage(content=f"Response from {provider}")
        mock_model_instance.ainvoke = AsyncMock(return_value=mock_response)
        
        # The mocked class should return our mocked instance
        mock_chat_class.return_value = mock_model_instance

        # 2. Set up the context and input for the graph
        model_string = f"{provider}:{model_name}"
        context = Context(model=model_string)
        graph_input = {"messages": [HumanMessage(content="Hello")]}

        # --- Act ---
        # Run the graph with the specified model provider
        result = await graph.ainvoke(
            graph_input, {"configurable": {"context": context}}
        )

        # --- Assert ---
        # 1. Verify the final output of the graph
        assert len(result["messages"]) == 2  # Human input + AI response
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert f"Response from {provider}" in final_message.content

        # 2. Verify that the correct model class was initialized once
        mock_chat_class.assert_called_once()
        
        # 3. Verify that the model instance was invoked
        mock_model_instance.ainvoke.assert_awaited_once()