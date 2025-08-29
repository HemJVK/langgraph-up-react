# Roadmap

This document outlines the planned upgrades and enhancements for the LangGraph ReAct Agent template.

## v0.3.0 - Productionization & Deployment (Planning)

> This phase focuses on making the agent easy to deploy and manage in production environments.

### LangGraph Platform Standalone Server
- **Containerization**: Provide an optimized `Dockerfile` for building the agent as a standalone service.
- **Docker Compose**: Include a `docker-compose.yml` for easy local and single-node deployments of the agent and any required services (e.g., a local Ollama instance).
- **Kubernetes Manifests**: Supply template Kubernetes manifests (Deployment, Service, ConfigMap) for deploying the agent to a K8s cluster.
- **Configuration Management**: Enhance the `Context` system to securely load configurations from environment variables and Kubernetes Secrets.

## v0.2.0 - Advanced Tooling & Evaluation (Planning)

> This phase will expand the agent's capabilities and introduce a formal framework for measuring its performance.

### Expanded Tool Library
- **File System Access**: Equip the agent with tools to read, write, and list files on a local file system (within a secure, sandboxed environment).
- **Database Tools**: Add tools for connecting to and querying SQL databases, allowing the agent to interact with structured data.
- **Advanced Agentic Tools**: Integrate more complex tools like `create_csv_agent` from LangChain to give the agent specialized, task-specific capabilities.

### Agent Evaluation Framework
- **LangSmith Evals Integration**: Implement an evaluation framework using LangSmith Evals to measure agent performance across various tasks.
- **Multi-Provider Benchmarking**: Create structured evaluation suites to compare the effectiveness, speed, and cost of different model providers (OpenAI, Groq, Gemini) on standardized benchmarks.
- **CI/CD Integration**: Integrate the evaluation tests into a CI/CD pipeline to automatically check for performance regressions.

## v0.1.0 - Multi-Provider Architecture & Core Enhancements (Completed)

> This version represents a complete architectural overhaul, moving from a single-provider implementation to a flexible, powerful, multi-provider platform.

### Model Provider Expansion
- **Multi-Provider Architecture**: Replaced the Qwen-specific implementation with a unified model factory (`create_model`) that supports **OpenAI, Groq, Google Gemini, and Ollama**.
- **Standardized Model Loading**: Implemented a simple and consistent `provider:model_name` format for specifying models.

### Dynamic & Enhanced Tooling
- **Local "BigTools"**: Integrated powerful, general-purpose tools from the LangChain ecosystem, including **Tavily Search** (web search) and **Python REPL** (code execution).
- **Remote Tool Extensibility**: Removed the hardcoded DeepWiki client and added support for the **OpenWebUI MCPO** protocol, allowing the agent to dynamically load tools from any compliant endpoint.

### Code Architecture & Prompt Refactoring
- **Modular Components**: Solidified the reusable `common/` module for context, models, tools, and prompts.
- **ReAct Prompt Engine**: Upgraded the system prompt to a robust ReAct (Reason-Action) format, essential for guiding the model's tool usage effectively.

### Comprehensive Test Suite Overhaul
- **Full Test Rewrite**: Replaced the entire test suite with new unit and integration tests covering the new architecture.
- **Mock-Based Testing**: The new suite uses extensive mocking and parametrization to ensure fast, reliable, and deterministic tests for all components and model providers.
- **ReAct Loop Validation**: Tests now validate the full Thought -> Action -> Observation cycle of the agent's logic.

### Development Experience
- **Simplified Configuration**: Streamlined the `.env` file to support the new providers.
- **Developer Tooling**: Maintained and validated the `make` commands for testing, linting, and running the development server.

---

Each version builds upon the previous foundation, progressively enhancing the agent's capabilities, developer experience, and deployment readiness.