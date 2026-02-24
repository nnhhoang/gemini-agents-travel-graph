# Gemini Travel Graph

Multi-agent travel planning system powered by Google Gemini and LangGraph orchestration. Includes a DynamoDB-backed AI conversation engine for Japanese tourism recommendations.

## Overview

Two modes:
- **Assistant mode** — AI tourism concierge chat (ConversationAgent + ContextBuilder). Uses user preferences, GPS, and time-of-day for personalized Japan travel recommendations.
- **Planner mode** — Full trip planning workflow (LangGraph StateGraph). Agents research destinations, search flights/accommodation, plan activities, and manage budgets.

## Key Features

```mermaid
mindmap
  root((Travel<br>Planning<br>System))
    (Multi-Agent Architecture)
      [Destination Research]
      [Flight Search]
      [Accommodation]
      [Transportation]
      [Activities]
    (Browser Automation)
      [Self-healing]
      [Parallel execution]
      [Data extraction]
    (Budget Management)
      [Cost tracking]
      [Value optimization]
      [Alternative options]
    (Personalization)
      [Preference analysis]
      [Custom recommendations]
    (Knowledge Storage)
      [Persistent memory]
      [Entity relationships]
```

## Technology Stack

- **AI**: [Google Gemini](https://ai.google.dev/) (google-genai SDK)
- **Orchestration**: [LangGraph v0.4+](https://github.com/langchain-ai/langgraph)
- **Browser Automation**: [Stagehand v2.0+](https://github.com/browserbase/stagehand) (built on Playwright)
- **Data Persistence**: [DynamoDB](https://aws.amazon.com/dynamodb/) (single-table design, boto3)
- **Research**: [Tavily API](https://tavily.com/), [Firecrawl](https://firecrawl.dev/)

See [Architecture & Requirements](docs/architecture-requirements.md) for full details.

## System Architecture

```mermaid
flowchart TD
    UI[User Interface Layer] --> OL

    subgraph OL[Orchestration Layer - LangGraph]
        SM[State Management]
        AWM[Agent Workflow Management]
        CP[Context Preservation]
    end

    OL --> DRA[Destination Research Agent]
    OL --> FSA[Flight Search Agent]
    OL --> ASA[Accommodation Search Agent]
    OL --> TPA[Transportation Planning Agent]
    OL --> APA[Activity Planning Agent]

    DRA & FSA & ASA & TPA & APA --> BAL[Browser Automation Layer]
    BAL --> BMA[Budget Management Agent]
    BMA --> KML[Knowledge & Memory Layer]
    KML --> PS[Persistent Storage - DynamoDB]

    classDef systemLayer fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef agent fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px

    class UI,OL,BAL,KML,PS systemLayer
    class DRA,FSA,ASA,TPA,APA,BMA agent
```

### Agent Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator as Orchestration Agent
    participant Destination as Destination Agent
    participant Flight as Flight Agent
    participant Hotel as Hotel Agent

    User->>Orchestrator: Travel Request
    Orchestrator->>Destination: Research Request
    Destination-->>Orchestrator: Destination Options

    par Flight & Hotel Search
        Orchestrator->>Flight: Search Flights
        Orchestrator->>Hotel: Search Accommodations
    end

    Flight-->>Orchestrator: Flight Options
    Hotel-->>Orchestrator: Accommodation Options
    Orchestrator->>User: Complete Itinerary

    Note over User,Hotel: Human feedback loop can interrupt at any stage
```

## Installation

```bash
# Install dependencies (production)
pip install -r requirements.txt

# Install dependencies (development)
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (GEMINI_API_KEY, DYNAMODB_TABLE_NAME)
```

### DynamoDB Setup

Single-table design. Configure via environment variables:

- `DYNAMODB_TABLE_NAME` — Table name (default: `travel-planner`)
- `AWS_REGION` — AWS region (default: `ap-northeast-1`)
- `DYNAMODB_ENDPOINT` — Set to `http://localhost:8000` for DynamoDB Local

```bash
docker run -p 8000:8000 amazon/dynamodb-local
```

The table is created automatically on first use.

## Usage

Interactive mode selection:

```bash
python -m travel_planner.main
```

Planner mode with query:

```bash
python -m travel_planner.main --mode planner --query "Visit Tokyo for a week" --origin "New York" --budget "3000-5000"
```

Assistant mode with preferences:

```bash
python -m travel_planner.main --mode assistant --preferences-file prefs.json
```

See `python -m travel_planner.main --help` for all options.

## Development

```bash
# Run tests
python -m pytest

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .
```

Dependencies are pinned in `requirements.txt` and `requirements-dev.txt`. Tool configs live in `pyproject.toml`.
