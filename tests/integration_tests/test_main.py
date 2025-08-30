"""Integration tests for the refactored LangGraph agent components."""

import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# FIX: Import the new, non-deprecated class
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool

from common.context import Context
from common.models import create_model
from common.tools import create_tools, get_mcpo_tools

from dotenv import load_dotenv

load_dotenv()

# FIX: Remove the global pytestmark to resolve warnings
# pytestmark = pytest.mark.asyncio


class TestContextConfiguration:
    """Tests the agent's configuration via the Context class."""

    def test_context_defaults(self) -> None:
        context = Context()
        assert context.model == "openai:gpt-4o-mini"
        assert context.max_search_results == 5
        assert hasattr(context, "mcpo_url")
        assert context.mcpo_url == ""

    @patch.dict(
        os.environ,
        {
            "MODEL": "groq:llama3-70b-8192",
            "MAX_SEARCH_RESULTS": "10",
            "MCPO_URL": "http://localhost:8080/mcp",
        },
        clear=True,
    )
    def test_context_initialization_from_env_vars(self) -> None:
        context = Context()
        assert context.model == "groq:llama3-70b-8192"
        assert str(context.max_search_results) == "10"
        assert context.mcpo_url == "http://localhost:8080/mcp"


@patch("common.models.ChatOpenAI", MagicMock())
@patch("common.models.ChatGroq", MagicMock())
@patch("common.models.ChatOllama", MagicMock())
@patch("common.models.ChatGoogleGenerativeAI", MagicMock())
class TestModelCreation:
    """Tests the model factory function."""

    # ... (This class is correct and needs no changes) ...
    def test_create_model_for_all_providers(self) -> None:
        # ...
        pass

    def test_create_unsupported_provider_raises_error(self) -> None:
        # ...
        pass

    def test_create_invalid_format_raises_error(self) -> None:
        # ...
        pass


class TestToolCreation:
    """Tests the creation and assembly of agent tools."""

    # FIX: Add the asyncio mark to the specific async test
    @pytest.mark.asyncio
    async def test_create_tools_local_only(self) -> None:
        """Test that create_tools returns only local tools when mcpo_url is not set."""
        context = Context(max_search_results=7, mcpo_url="")
        tools = await create_tools(context)

        assert len(tools) == 6
        # FIX: Check for the new TavilySearch class
        assert any(isinstance(t, TavilySearch) for t in tools)

    @pytest.mark.asyncio
    @patch("common.tools.get_mcpo_tools", new_callable=AsyncMock)
    async def test_create_tools_with_mcpo(self, mock_get_mcpo_tools: AsyncMock) -> None:
        # ... (This test is correct and needs no changes) ...
        pass

    @pytest.mark.asyncio
    async def test_get_mcpo_tools_no_url(self) -> None:
        # ... (This test is correct and needs no changes) ...
        pass
