"""
Comprehensive unit tests for the agent's tool creation and management module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
# FIX: Import the new, non-deprecated class
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool

from common.context import Context
from common.tools import create_tools, get_mcpo_tools

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- FIX: Add a fixture to reset the global state before each test ---
@pytest.fixture(autouse=True)
def reset_mcp_client_cache():
    """
    This fixture automatically runs before each test in this file.
    It resets the global _mcp_client to prevent test pollution.
    """
    from common import tools
    tools._mcp_client = None
    yield
# --------------------------------------------------------------------

class TestLocalToolCreation:
    """Tests the creation and configuration of local tools."""

    async def test_create_tools_local_only(self) -> None:
        context = Context(max_search_results=10, mcpo_url="")
        tools = await create_tools(context)

        assert len(tools) == 6
        # FIX: Check for the new class name
        assert any(isinstance(t, TavilySearch) for t in tools)
        assert any(isinstance(t, PythonREPLTool) for t in tools)

        search_tool = next(t for t in tools if isinstance(t, TavilySearch))
        assert search_tool.max_results == 10


class TestMCPOIntegration:
    """Tests the integration with the OpenWebUI MCPO client."""

    @patch("common.tools.MultiServerMCPClient")
    async def test_get_mcpo_tools_success(self, mock_mcp_client_class: MagicMock) -> None:
        mock_tool = MagicMock(name="RemoteTool")
        mock_client_instance = MagicMock()
        mock_client_instance.get_tools = AsyncMock(return_value=[mock_tool])
        mock_mcp_client_class.return_value = mock_client_instance
        mcpo_url = "http://fake-url:8080/mcp"

        tools = await get_mcpo_tools(mcpo_url)

        assert tools == [mock_tool]
        mock_mcp_client_class.assert_called_once_with(
            {"openwebui_mcpo": {"url": mcpo_url, "transport": "sse"}}
        )

    async def test_get_mcpo_tools_with_empty_url(self) -> None:
        assert await get_mcpo_tools("") == []
        assert await get_mcpo_tools(None) == []

    @patch("common.tools.MultiServerMCPClient")
    async def test_get_mcpo_tools_client_initialization_failure(
        self, mock_mcp_client_class: MagicMock
    ) -> None:
        # This test will now pass because the fixture resets the global client.
        mock_mcp_client_class.side_effect = ConnectionError("Connection failed")
        tools = await get_mcpo_tools("http://some-url")
        assert tools == []


class TestCombinedToolCreation:
    """Tests the combination of local and remote tools."""

    @patch("common.tools.get_mcpo_tools", new_callable=AsyncMock)
    async def test_create_tools_combines_local_and_remote(
        self, mock_get_mcpo_tools: AsyncMock
    ) -> None:
        mock_remote_tool = MagicMock(name="RemoteSearch")
        mock_get_mcpo_tools.return_value = [mock_remote_tool]
        context = Context(mcpo_url="http://test-server/mcp")

        all_tools = await create_tools(context)

        mock_get_mcpo_tools.assert_awaited_once_with("http://test-server/mcp")
        assert len(all_tools) == 7
        # FIX: Check for the new class name
        assert any(isinstance(t, TavilySearch) for t in all_tools)
        assert any(isinstance(t, PythonREPLTool) for t in all_tools)
        assert mock_remote_tool in all_tools