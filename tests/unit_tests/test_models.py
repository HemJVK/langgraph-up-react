"""
Unit tests for the custom model integration module.

This file tests the model factory (`create_model`), the helper function to
list supported models, and the `load_chat_model` utility.
"""

from unittest.mock import patch, MagicMock
import pytest

from dotenv import load_dotenv

load_dotenv()

from common.models import create_model, get_supported_models
from common.utils import load_chat_model

# A list of model providers to test.
# When a new provider is added, its info should be added here for automatic testing.
SUPPORTED_PROVIDERS_CONFIG = [
    ("openai", "gpt-4o-mini", "common.models.ChatOpenAI"),
    ("groq", "llama3-70b-8192", "common.models.ChatGroq"),
    ("ollama", "llama3", "common.models.ChatOllama"),
    ("gemini", "gemini-1.5-pro-latest", "common.models.ChatGoogleGenerativeAI"),
]


class TestModelFactory:
    """Tests the create_model factory function."""

    def test_get_supported_models(self) -> None:
        """
        Verify that get_supported_models returns a non-empty list of strings.
        """
        models = get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Check that the format seems correct for at least one entry
        assert ":" in models[0]

    @pytest.mark.parametrize("provider, model_name, mock_path", SUPPORTED_PROVIDERS_CONFIG)
    def test_create_model_dispatches_correctly(
        self, provider: str, model_name: str, mock_path: str
    ) -> None:
        """
        Test that create_model calls the correct LangChain class for each provider.
        This is a parametrized test that runs for every entry in SUPPORTED_PROVIDERS_CONFIG.
        """
        with patch(mock_path) as mock_chat_class:
            model_string = f"{provider}:{model_name}"
            create_model(model_string)

            # Assert that the correct class was instantiated
            mock_chat_class.assert_called_once()
            call_args = mock_chat_class.call_args[1]

            # Groq's LangChain integration uses 'model_name' instead of 'model'
            if provider == "groq":
                assert call_args["model_name"] == model_name
            else:
                assert call_args["model"] == model_name

    def test_create_model_passes_extra_kwargs(self) -> None:
        """
        Verify that additional keyword arguments are passed to the model's constructor.
        """
        with patch("common.models.ChatOpenAI") as mock_chat_openai:
            create_model("openai:gpt-4o", temperature=0.5, max_tokens=100)
            mock_chat_openai.assert_called_once_with(
                model="gpt-4o", temperature=0.5, max_tokens=100
            )


class TestLoadChatModelUtil:
    """Tests the load_chat_model utility function."""

    @patch("common.utils.create_model")
    def test_load_chat_model_is_a_wrapper_for_create_model(self, mock_create_model: MagicMock) -> None:
        """
        Verify that load_chat_model simply calls create_model.
        This confirms we have successfully centralized our model creation logic.
        """
        # Arrange
        mock_model_instance = MagicMock()
        mock_create_model.return_value = mock_model_instance
        model_string = "openai:gpt-4o-mini"

        # Act
        result = load_chat_model(model_string)

        # Assert
        assert result is mock_model_instance
        mock_create_model.assert_called_once_with(model_string)