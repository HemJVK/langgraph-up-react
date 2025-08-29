"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

from dotenv import load_dotenv
load_dotenv()

# --- FIX: Import the Context class ---
from common.context import Context
# -------------------------------------

@dataclass
class InputState:
    """Defines the input state for the agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    # --- FIX: Add context directly to the state ---
    context: Context = field(default_factory=Context)
    # --------------------------------------------

@dataclass
class State(InputState):
    """Represents the complete state of the agent."""

    is_last_step: IsLastStep = field(default=False)