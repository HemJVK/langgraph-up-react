"""Tests for error handling and edge cases in the agent's components."""

import os
from unittest.mock import patch

import pytest

from common.context import Context
from common.models import create_model

# A list of invalid model strings to test the model factory's error handling.
INVALID_MODEL_STRINGS = [
    "invalid-format-no-separator", # Fails first check
    "",                            # Fails first check
    "provider:",                   # Should fail second check
    ":model-name",                 # Should fail second check
]


class TestModelErrorHandling:
    """Tests error handling and edge cases for the `create_model` factory."""

    @pytest.mark.parametrize("invalid_string", INVALID_MODEL_STRINGS)
    def test_invalid_model_string_format_raises_error(self, invalid_string: str) -> None:
        """Test that various invalid model string formats raise a ValueError."""
        # This test will now pass for all cases because our improved create_model
        # correctly identifies all these strings as invalid formats.
        with pytest.raises(ValueError, match="Invalid model string"):
            create_model(invalid_string)

    def test_unsupported_model_provider_raises_error(self) -> None:
        """Test that an unsupported provider in a correctly formatted string raises a ValueError."""
        with pytest.raises(ValueError, match="Unsupported model provider: 'unknown'"):
            create_model("unknown:some-model")

    @patch("common.models.ChatOpenAI")
    def test_model_initialization_failure_propagates(self, mock_chat_openai: patch) -> None:
        """Test that an exception during a model's initialization is propagated up."""
        mock_chat_openai.side_effect = RuntimeError("Failed to connect to API")

        with pytest.raises(RuntimeError, match="Failed to connect to API"):
            create_model("openai:gpt-4o")

    def test_passing_none_as_model_string_raises_error(self) -> None:
        """Test that passing None to create_model raises an error."""
        with pytest.raises(Exception):
            create_model(None) # type: ignore


class TestContextEdgeCases:
    """Tests edge cases and boundary conditions for the Context class."""

    @pytest.mark.parametrize("results_value", [0, -1, 10000])
    def test_context_handles_boundary_max_search_results(self, results_value: int) -> None:
        """Test that Context accepts boundary values for max_search_results."""
        context = Context(max_search_results=results_value)
        assert context.max_search_results == results_value

    def test_context_handles_special_characters_in_prompt(self) -> None:
        """Test that Context correctly handles special characters in the system prompt."""
        special_prompt = "Prompt with émojis 🤖, spëcial chärs & symbols!"
        context = Context(system_prompt=special_prompt)
        assert context.system_prompt == special_prompt

    def test_context_handles_very_long_system_prompt(self) -> None:
        """Test that Context handles an extremely long system prompt string."""
        long_prompt = "long string " * 2000
        context = Context(system_prompt=long_prompt)
        assert context.system_prompt == long_prompt

    @patch.dict(os.environ, {"MODEL": "", "SYSTEM_PROMPT": ""}, clear=True)
    def test_context_handles_empty_environment_variables(self) -> None:
        """Test that Context handles empty strings from environment variables."""
        context = Context()
        # The __post_init__ should set the attributes to the empty string from the env var.
        assert context.model == ""
        assert context.system_prompt == ""