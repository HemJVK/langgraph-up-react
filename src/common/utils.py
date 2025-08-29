"""Utility & helper functions."""

from typing import Any, Dict, List, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .models import create_model

from dotenv import load_dotenv
load_dotenv()


def get_message_text(msg: BaseMessage) -> str:
    """
    Get the text content from a LangChain message object.

    This helper function robustly extracts text from a message's content,
    which can be a simple string or a list of content blocks (e.g., for
    multi-modal inputs).

    Args:
        msg: The LangChain message object.

    Returns:
        The extracted text content as a string.
    """
    content = msg.content
    if isinstance(content, str):
        return content

    # Handle list-based content (common in multi-modal models)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts).strip()

    return ""


def load_chat_model(model_string: str) -> BaseChatModel:
    """
    Load a chat model using the centralized model factory.

    This function acts as a wrapper around the `create_model` function,
    providing a consistent way to load any supported chat model using a
    'provider:model_name' string.

    Args:
        model_string: A string identifying the model and its provider
                      (e.g., "openai:gpt-4o-mini", "groq:llama3-70b-8192").

    Returns:
        An initialized instance of the specified chat model.
    """
    # Delegate the model creation to our new, centralized factory function.
    return create_model(model_string)
