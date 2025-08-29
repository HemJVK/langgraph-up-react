"""Pytest configuration and shared fixtures for the agent tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# A list of API keys for the newly supported services.
# Tests that require a live model will be skipped if the relevant key is not found.
REQUIRED_API_KEYS = [
    "TAVILY_API_KEY",      # For the primary search tool
    "OPENAI_API_KEY",      # For OpenAI/ChatGPT models
    "GROQ_API_KEY",        # For Groq models
    "GOOGLE_API_KEY",      # For Google Gemini models
]

@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """
    Loads environment variables from a .env file at the project root.
    It also checks for the presence of required API keys and skips tests if
    any are missing, preventing test failures due to configuration issues.
    """
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    missing_keys = [key for key in REQUIRED_API_KEYS if not os.getenv(key)]

    if missing_keys:
        # This will cause tests that rely on these keys to be skipped,
        # rather than fail.
        pytest.skip(f"Skipping live integration tests. Missing env vars: {missing_keys}")


class TestHelpers:
    """A collection of helper methods for writing cleaner tests."""

    @staticmethod
    def assert_valid_response(
        messages: list, expected_content: str | None = None, min_messages: int = 2
    ) -> None:
        """Asserts that a list of messages from a graph run has a valid structure."""
        assert isinstance(messages, list)
        assert len(messages) >= min_messages

        final_message = messages[-1]
        # Handle both dict and object-based message formats
        content_attr = getattr(final_message, "content", None)
        content_item = final_message.get("content", "") if isinstance(final_message, dict) else ""
        final_content = content_attr or content_item

        assert final_content is not None, "Final message should have content."

        if expected_content:
            assert expected_content.lower() in str(final_content).lower()

    @staticmethod
    def assert_tool_usage(messages: list, tool_name: str) -> None:
        """Asserts that a specific tool was called during the graph run."""
        tool_call_found = False
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == tool_name:
                        tool_call_found = True
                        break
            if tool_call_found:
                break
        
        assert tool_call_found, f"Expected tool '{tool_name}' to be used, but it was not found."


@pytest.fixture
def helpers() -> TestHelpers:
    """Provides an instance of the TestHelpers class to tests."""
    return TestHelpers()