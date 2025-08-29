"""Tools for the LangGraph agent, including local and remote MCPO tools."""

import logging
from typing import List, Any, Optional

from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
# FIX: Import the new, non-deprecated class from the correct package
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

from .context import Context
from .vector_store import add_document, search_documents

load_dotenv()

logger = logging.getLogger(__name__)

# FIX: Use the new class name and max_results argument
web_search = TavilySearch(max_results=5)

# Global cache for the MCP client to avoid re-initialization.
_mcp_client: Optional[MultiServerMCPClient] = None


@tool
def add_document_tool(document: str) -> str:
    """Adds a document to the vector store."""
    add_document(document)
    return f"Document added to the vector store: {document}"


@tool
def search_documents_tool(query: str, k: int = 4) -> list[str]:
    """Searches for similar documents in the vector store."""
    return search_documents(query, k)


async def get_mcpo_tools(mcpo_url: Optional[str]) -> List[Any]:
    """
    Fetches tools from an OpenWebUI MCPO server.
    ... (docstring is correct) ...
    """
    global _mcp_client
    if not mcpo_url:
        return []

    server_name = "openwebui_mcpo"
    server_config = {
        server_name: {"url": mcpo_url, "transport": "sse"}
    }

    try:
        if _mcp_client is None:
            _mcp_client = MultiServerMCPClient(server_config)
            logger.info(f"Initialized MCP client for server: {server_name}")

        tools = await _mcp_client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from MCPO server at {mcpo_url}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load tools from MCPO server at {mcpo_url}: {e}")
        return []


async def create_tools(context: Context) -> List[Any]:
    """
    Create and return a list of tools for the agent.
    ... (docstring is correct) ...
    """
    # 1. Initialize local tools from LangChain.
    # FIX: Use the new class name
    search_tool = TavilySearch(max_results=context.max_search_results)
    python_repl_tool = PythonREPLTool()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    local_tools = [search_tool, python_repl_tool, wikipedia_tool, arxiv_tool, add_document_tool, search_documents_tool]
    logger.info(f"Loaded {len(local_tools)} local tools (Search, Python REPL, Wikipedia, Arxiv, Vector Store).")

    # 2. Fetch remote tools from the OpenWebUI MCPO server, if configured.
    remote_tools = []
    mcpo_url = getattr(context, 'mcpo_url', None)
    if mcpo_url:
        remote_tools = await get_mcpo_tools(mcpo_url)

    # 3. Combine and return all tools.
    all_tools = local_tools + remote_tools
    logger.info(f"Total tools created: {len(all_tools)}.")
    return all_tools