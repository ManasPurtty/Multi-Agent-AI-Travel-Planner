# Wayfarer — AI Travel Planner

Wayfarer turns a travel request into a researched draft itinerary. A LangGraph
supervisor routes work to flight, hotel, weather, and budget specialists; you
review the draft before the app creates a polished final plan.

## What you need

- Python 3.13 (required by the AviationStack MCP package)
- API keys for Groq, Tavily, AviationStack, and OpenWeatherMap
- The AviationStack MCP server source in `aviationstack-mcp/`
- PostgreSQL is optional. Without it, conversation checkpoints are kept in
  memory and are lost when the app stops.

## Quick start (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your API keys. If you want checkpoints to survive app
restarts, create a PostgreSQL database and set `DATABASE_URL`, for example:

```text
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/langgraph_memory_demo
```

The default in-memory checkpoint store supports draft approval for the life of
the running app. PostgreSQL enables durable checkpoint storage.

### AviationStack MCP source

The MCP server is kept outside this repository. Clone and install its source into
the project directory so the AviationStack agent can start it:

```powershell
git clone https://github.com/Pradumnasaraf/aviationstack-mcp.git
python -m pip install .\aviationstack-mcp
```

The app starts its MCP server with the current Python interpreter. Keep this
directory out of Git; deployment installs it during the build.

### Start the app

```powershell
streamlit run frontend.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Choose an
example prompt or describe a trip, review the draft, and submit your approval
or requested changes.

## Environment variables

| Variable | Used for |
| --- | --- |
| `GROQ_API_KEY` | Supervisor and itinerary agents (required) |
| `GROQ_MODEL` | Optional model override; defaults to `openai/gpt-oss-120b` |
| `TAVILY_API_KEY` | Web and accommodation research |
| `AVIATION_STACK_API_KEY` | Airport and airline reference data |
| `OPENWEATHER_API_KEY` | Current weather and forecast |
| `DATABASE_URL` | Optional PostgreSQL checkpoint storage |

## Deploy and share

This repository includes a Render Blueprint in `render.yaml`. To deploy:

1. Push this project directory to a GitHub repository. Do not commit `.env`;
   it is ignored by Git.
2. In Render, choose **New** → **Blueprint**, connect the GitHub repository,
   and select the directory containing `render.yaml` if it is not at the
   repository root.
3. In the Blueprint setup, supply `GROQ_API_KEY`, `TAVILY_API_KEY`,
   `AVIATION_STACK_API_KEY`, and `OPENWEATHER_API_KEY`. These are stored as
   Render environment variables, not in the repository.
4. Deploy. When the service is live, Render provides a public `onrender.com`
   URL that you can share.

The Blueprint selects Python 3.13, installs the root requirements and the
AviationStack MCP package, and starts Streamlit on Render's assigned port.
`GROQ_MODEL` has a default and can be overridden in Render. `DATABASE_URL` is
optional; without it, checkpoint state is held in memory and can be lost when
the service restarts. Anyone with the public URL can use the app and consume
your API quotas, so monitor provider usage.

## Notes

- API-backed research requires valid keys and network access.
- Plans are recommendations. Verify current prices, availability, schedules,
  and entry requirements before booking.
- PostgreSQL checkpointing stores application state. Protect database access
  and credentials accordingly.

## Background

This project builds on the
[LangGraph AI Travel Planning System](https://github.com/codewithaarohi/AI-Travel-Planning-System-using-LangGraph)
and its [MCP edition](https://github.com/codewithaarohi/AI-Travel-Planning-App-using-LangGraph-and-MCP).
