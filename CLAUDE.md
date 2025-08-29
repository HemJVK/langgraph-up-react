```markdown
# AI Assistant Guide (CLAUDE.md)

This file provides guidance to AI assistants when working with code in this repository.

## Project Overview

This is a powerful, multi-provider LangGraph ReAct (Reasoning and Action) Agent template. It implements a modular, iterative reasoning agent that leverages the LangGraph framework to process user queries through a cycle of thought, action, and observation.

## Architecture

The core architecture follows a modular, stateful graph pattern:

- **Common Module**: Shared components in `src/common/` provide reusable, high-level functionality.
- **State Management**: Uses `State` and `InputState` dataclasses (defined in `src/react_agent/state.py`) to track the conversation and execution state.
- **Graph Structure**: The main graph is defined in `src/react_agent/graph.py` with two primary nodes:
  - `call_model`: The "brain" of the agent, responsible for LLM reasoning and tool selection.
  - `tools`: The "hands" of the agent, executing the tools selected by the model.
- **Execution Flow**: `call_model` → conditional routing → `tools` (if a tool is called) or `__end__` (to provide a final answer). The `tools` node always routes back to `call_model`, creating the ReAct loop.
- **Context System**: A runtime context defined in `src/common/context.py` provides dynamic configuration for the model, system prompts, search parameters, and remote tool endpoints.
- **Dynamic Tools**: The agent is equipped with a powerful set of local tools (e.g., web search, Python interpreter) and can be extended with remote tools via an OpenWebUI MCPO server.

## Development Commands

### Testing
```bash
# Run specific test types
make test                    # Run unit and integration tests (default)
make test_unit               # Run unit tests only
make test_integration        # Run integration tests only
make test_e2e                # Run e2e tests only (requires running LangGraph server)
make test_all                # Run all tests (unit + integration + e2e)
```

### Code Quality
```bash
make lint                   # Run linters (ruff + mypy)
make format                 # Auto-format code with ruff
make lint_package           # Lint only src/ directory
make lint_tests             # Lint only tests/ directory
```

### Development Server
```bash
make dev                    # Start LangGraph development server
make dev_ui                 # Start LangGraph development server with UI
```

### Environment Setup
- Copy `.env.example` to `.env` and configure API keys.
- **Required**: `TAVILY_API_KEY` for web search functionality.
- **Model Providers (fill in at least one)**: `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`.
- **Optional (for remote tools)**: `MCPO_URL`: The URL of an OpenWebUI MCPO server to fetch additional tools from.
- **Default Model**: The agent uses `openai:gpt-4o-mini` by default. This can be changed via the `MODEL` environment variable (e.g., `MODEL=groq:llama3-70b-8192`).


## Key Files and Their Purposes

### Core Agent Files
- `src/react_agent/graph.py`: Main graph definition and ReAct loop implementation.
- `src/react_agent/state.py`: State management classes for tracking messages and graph state.

### Common Module (Shared Components)
- `src/common/context.py`: Defines the runtime configuration (`Context`) for the agent.
- `src/common/tools.py`: The tool factory. Creates and assembles the agent's tools, including local LangChain tools and remote MCPO tools.
- `src/common/models.py`: The model factory. Instantiates the correct language model (OpenAI, Groq, Gemini, Ollama) based on a `provider:model_name` string.
- `src/common/prompts.py`: Contains the ReAct system prompt template.
- `src/common/utils.py`: Shared utility functions, primarily the `load_chat_model` wrapper.

### Configuration
- `langgraph.json`: LangGraph Studio configuration pointing to the main graph.
- `.env`: Environment variables for API keys and configuration.

## LangGraph Studio Integration

This project works seamlessly with LangGraph Studio. The `langgraph.json` config file defines:
- Graph entry point: `./src/react_agent/graph.py:graph`
- Environment file: `.env`
- Dependencies: current directory (`.`)

## Tool Integration

This project features a robust and extensible tool system.

### Local Tools ("BigTools")
The agent comes pre-configured with powerful, general-purpose tools from the LangChain ecosystem:
- **Tavily Search**: A best-in-class search engine designed for LLMs.
- **Python REPL**: A code interpreter that allows the agent to perform calculations, run algorithms, and manipulate data.

### Remote Tools (OpenWebUI MCPO)
The agent can dynamically load additional tools from any OpenWebUI instance that exposes an MCPO (Multi-tool Calling Protocol Orchestrator) endpoint.
- **Configuration**: Simply set the `MCPO_URL` environment variable to the server's address.
- **Dynamic Loading**: Tools are fetched at runtime and seamlessly added to the agent's capabilities.
- **Extensibility**: This allows you to add custom or specialized tools to your agent without modifying its core code.

## Python Configuration

- Python requirement: `>=3.11,<4.0`
- Main dependencies: `langgraph`, `langchain`, `langchain-openai`, `langchain-groq`, `langchain-google-genai`, `langchain-community`, `tavily-python`.
- Development tools: `mypy`, `ruff`, `pytest`.

## Development Guidelines

- **Code Quality**: Always run `make lint` and `make format` before committing changes.
- **Testing**: Ensure that any new functionality is covered by unit or integration tests. The test suite uses mocking extensively to ensure fast and reliable execution.
```