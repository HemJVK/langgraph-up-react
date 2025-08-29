"""Tests for the agent's Context configuration class."""

import os
from unittest.mock import patch

import pytest

from common.context import Context

from dotenv import load_dotenv

load_dotenv()

# A representative list of new supported models for testing parametrization.
SUPPORTED_MODELS = [
    "openai:gpt-4o-mini",
    "groq:llama3-8b-8192",
    "ollama:llama3",
    "gemini:gemini-1.5-pro-latest",
]


class TestContextConfiguration:
    """Test suite for the Context configuration class."""

    def test_context_defaults(self) -> None:
        """Test that Context initializes with the correct default values."""
        context = Context()
        assert context.model == "openai:gpt-4o-mini"
        assert context.max_search_results == 5
        assert context.mcpo_url == ""
        # Check that a default system prompt exists and is not empty
        assert context.system_prompt and isinstance(context.system_prompt, str)

    def test_context_explicit_initialization(self) -> None:
        """Test that all Context fields can be set explicitly in the constructor."""
        custom_prompt = "You are a test assistant."
        context = Context(
            model="groq:llama3-70b-8192",
            system_prompt=custom_prompt,
            max_search_results=10,
            mcpo_url="http://localhost:8080/mcp",
        )
        assert context.model == "groq:llama3-70b-8192"
        assert context.system_prompt == custom_prompt
        assert context.max_search_results == 10
        assert context.mcpo_url == "http://localhost:8080/mcp"

    @patch.dict(
        os.environ,
        {
            "MODEL": "gemini:gemini-1.5-flash-latest",
            "SYSTEM_PROMPT": "Prompt from environment.",
            "MAX_SEARCH_RESULTS": "15",
            "MCPO_URL": "http://env-url:8080/mcp",
        },
        clear=True,
    )
    def test_context_loads_from_environment_variables(self) -> None:
        """Test that all Context fields can be loaded from environment variables."""
        context = Context()
        assert context.model == "gemini:gemini-1.5-flash-latest"
        assert context.system_prompt == "Prompt from environment."
        # Environment variables are loaded as strings
        assert str(context.max_search_results) == "15"
        assert context.mcpo_url == "http://env-url:8080/mcp"

    @patch.dict(os.environ, {"MODEL": "groq:llama3-8b-8192"}, clear=True)
    def test_explicit_parameters_override_environment_variables(self) -> None:
        """Test that parameters passed to the constructor take precedence over env vars."""
        # The environment variable is set to a Groq model.
        # We explicitly pass an OpenAI model to the constructor.
        context = Context(model="openai:gpt-4o")
        assert context.model == "openai:gpt-4o"

    @pytest.mark.parametrize("model_string", SUPPORTED_MODELS)
    def test_colon_separator_format_is_valid(self, model_string: str) -> None:
        """Test that the 'provider:model' format is handled correctly."""
        context = Context(model=model_string)
        assert context.model == model_string
        assert ":" in context.model, "Model string should use the colon separator."
