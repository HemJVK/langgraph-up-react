"""
Centralized test data and constants for the agent's test suite.

This file provides a single source of truth for frequently used strings and
configurations in tests, such as model names and test prompts. This makes
the test suite easier to read and maintain.
"""


class TestModels:
    """A collection of representative model strings for the newly supported providers."""

    # OpenAI Models
    OPENAI_GPT4o_MINI = "openai:gpt-4o-mini"
    OPENAI_GPT4o = "openai:gpt-4o"

    # Groq Models
    GROQ_LLAMA3_8B = "groq:llama3-8b-8192"
    GROQ_LLAMA3_70B = "groq:llama3-70b-8192"

    # Ollama Models (example names, depends on local setup)
    OLLAMA_LLAMA3 = "ollama:llama3"
    OLLAMA_PHI3 = "ollama:phi3"

    # Google Gemini Models
    GEMINI_1_5_PRO = "gemini:gemini-1.5-pro-latest"
    GEMINI_1_5_FLASH = "gemini:gemini-1.5-flash-latest"


class TestQuestions:
    """
    A collection of common test prompts and expected outcomes for integration tests.
    """

    # A simple question that a capable model should be able to answer without tools.
    SIMPLE_KNOWLEDGE = {
        "question": "What is the capital of France?",
        "expected_in_response": "Paris",
        "requires_tools": False,
    }

    # A question that requires current information, necessitating a web search.
    TOOL_USAGE_KNOWLEDGE = {
        "question": "Who won the last Super Bowl?",
        "expected_in_response": "Chiefs",  # As of early 2024
        "requires_tools": True,
        "tool_to_use": "tavily_search_results_json",
    }

    # A question that requires calculation, necessitating the Python REPL tool.
    TOOL_USAGE_CALCULATION = {
        "question": "What is 3 to the power of 5?",
        "expected_in_response": "243",
        "requires_tools": True,
        "tool_to_use": "python_repl_tool",
    }