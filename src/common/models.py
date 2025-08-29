"""Custom model integrations for the LangGraph agent."""

import os
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

from dotenv import load_dotenv
load_dotenv()

def create_model(model_string: str, **kwargs: Any) -> BaseChatModel:
    """
    Create a chat model instance based on a provider-prefixed model string.
    ... (docstring is correct) ...
    """
    if ":" not in model_string:
        raise ValueError(
            f"Invalid model string: '{model_string}'. "
            "Expected format is 'provider:model_name'."
        )

    provider, model_name = model_string.split(":", 1)

    # --- THIS IS THE CRUCIAL FIX ---
    # Add validation to ensure both parts of the string are non-empty.
    if not provider or not model_name:
        raise ValueError(
            f"Invalid model string: '{model_string}'. "
            "Both provider and model_name must be specified."
        )
    # --------------------------------

    model_kwargs = {"model": model_name, **kwargs}

    if provider == "openai":
        return ChatOpenAI(**model_kwargs)

    elif provider == "groq":
        model_kwargs["model_name"] = model_name
        del model_kwargs["model"]
        return ChatGroq(**model_kwargs)

    elif provider == "ollama":
        return ChatOllama(**model_kwargs)

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(**model_kwargs)

    else:
        raise ValueError(
            f"Unsupported model provider: '{provider}'. "
            "Supported providers are 'openai', 'groq', 'ollama', 'gemini'."
        )


def get_supported_models() -> List[str]:
    """
    Get a list of example supported models for each provider.
    ... (docstring is correct) ...
    """
    return [
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "openai:gpt-4-turbo",
        "openai:gpt-3.5-turbo",
        "groq:llama3-70b-8192",
        "groq:llama3-8b-8192",
        "groq:mixtral-8x7b-32768",
        "ollama:llama3",
        "ollama:phi3",
        "ollama:mistral",
        "gemini:gemini-1.5-pro-latest",
        "gemini:gemini-1.5-flash-latest",
    ]
