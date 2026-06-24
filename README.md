# F.A.S.T — Financial Analysis and Stock Trading Analysis

Real-time stock analysis platform combining **market data**, **sentiment**, **forecasting**, **technical indicators**, **Google Trends**, **X (Twitter) social streams**, and production-grade **Agentic AI**, **RAG**, **FastAPI**, and **MCP** integration.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Features by section](#features-by-section)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Run the application](#run-the-application)
7. [Optional services](#optional-services)
8. [API & MCP](#api--mcp)
9. [Troubleshooting](#troubleshooting)
10. [Disclaimer](#disclaimer)

---

## Project structure

```
Real-Time-Stock-Analysis-using-Time-Series-and-Sentiment-Prediction/
│
├── app.py                      # Streamlit entry point (main UI)
├── run_app.py                  # Launcher: pip install + streamlit run
├── run.bat                     # Windows shortcut to run_app flow
├── start_kafka.bat             # Starts Kafka (calls scripts/start_kafka.bat)
├── requirements.txt
├── .env.example                # Copy to .env and fill in keys
│
├── config/                     # App configuration
│   ├── settings.py             # .env → Pydantic settings (LLM, X, Kafka, paths)
│   └── paths.py                # Resolved paths with legacy folder fallbacks
│
├── ui/                         # Streamlit theme & navigation
│   └── theme.py                # CSS, sidebar nav, hero cards, metrics
│
├── views/                      # Streamlit page modules (extracted from app.py)
│   ├── ai_analyst.py           # Agentic AI chat UI
│   ├── news_sentiment_ui.py    # News sentiment charts & tables
│   ├── overview_dashboard.py   # Overview / market dashboard (Plotly)
│   └── x_social_trends.py      # X + Kafka social trends page
│
├── services/                   # Business logic & external APIs
│   ├── market.py               # Yahoo Finance OHLCV
│   ├── news.py                 # Finviz headlines + VADER sentiment
│   ├── fundamentals.py         # Finviz company snapshot
│   ├── forecast.py             # Prophet / polynomial forecasts
│   ├── technicals.py             # SMA, EMA, MACD, signals
│   ├── indicators.py           # RSI, Bollinger Bands
│   ├── google_trends.py        # Google Trends news/web interest + Google News RSS
│   ├── x_twitter.py              # X API client (OAuth1, tweet search)
│   └── x_trends_store.py       # Local tweet aggregate store (Kafka consumer output)
│
├── streaming/                  # Real-time X tweet pipeline
│   ├── kafka_config.py         # Producer/consumer helpers + health checks
│   ├── x_producer.py           # Polls X API → Kafka topic
│   └── x_consumer.py           # Kafka → local trend store
│
├── agents/                     # LLM agent orchestration
│   ├── orchestrator.py         # Tool-calling agent loop
│   ├── tools.py                # 7 financial analysis tools
│   └── prompts.py              # System prompts
│
├── rag/                        # Retrieval-augmented generation
│   ├── ingest.py               # Index transcripts + news into Chroma
│   ├── store.py                # ChromaDB vector store
│   └── retriever.py            # Semantic search over earnings corpus
│
├── api/                        # FastAPI REST backend
│   ├── main.py                 # /chat, /analyze, /rag/query, /health
│   └── models.py               # Request/response schemas
│
├── mcp_server/                 # Model Context Protocol (Cursor / Claude Desktop)
│   └── server.py
│
├── scripts/                    # CLI utilities
│   ├── ingest_rag.py           # Index earnings transcripts
│   ├── run_x_kafka_stream.py   # Standalone X → Kafka pipeline
│   ├── validate_x_credentials.py
│   └── start_kafka.bat
│
├── infra/                      # Infrastructure definitions
│   └── docker-compose.kafka.yml
│
├── data/                       # Application data (git: datasets + transcripts)
│   ├── datasets/SP500.csv      # S&P 500 ticker list
│   ├── transcripts/            # Earnings call transcripts (RAG source)
│   ├── chroma/                 # Vector DB (generated, gitignored)
│   └── x_trends/               # X tweet aggregates (generated, gitignored)
│
├── assets/images/              # Static images (architecture diagram, etc.)
│
└── .streamlit/config.toml      # Streamlit server options
```

**Legacy folders** (`inference-data/`, `Datasets/`, root `ui_theme.py`) still work via `config/paths.py` fallbacks but new installs should use `data/` and `ui/`.

---

## Features by section

### Dashboard
Live market overview, ticker search, price charts, and quick stats for the selected symbol (Yahoo Finance).

### Agentic AI (`views/ai_analyst.py`)
LLM-powered analyst with **tool calling** (quotes, news, fundamentals, technicals, forecasts, RAG). Supports **OpenAI**, **Groq**, and **Gemini** via `.env`.

### Overview
Project data sources, AWS architecture diagram, and an open-source **Plotly market dashboard** (sector map, watchlist, index trends).

### News & sentiment
Finviz headlines scored with **VADER**; donut chart, histogram, timeline, and styled news table.

### Company profile
Finviz snapshot: sector, P/E, market cap, income, and related fundamentals.

### Technicals
SMA/EMA, MACD, **RSI**, **Bollinger Bands**, and trading signal summary.

### Search trends
**Google Trends** news (then web) search interest with forecast; related headlines from **Google News RSS**. Falls back to **Finviz headline activity** if Trends is rate-limited.

### Social trends (`views/x_social_trends.py`)
Real-time **X (Twitter)** tweets for stock cashtags → **Kafka** topic → consumer → local store. **VADER sentiment**, volume charts, Prophet forecast. Works in **direct ingest mode** without Kafka.

### Meeting notes
Earnings transcript **summarization** (extractive or **RAG + AI** over `data/transcripts/`).

### Forecast
S&P 500 picker with historical prices and forward **Prophet** projection.

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| **Python 3.10+** | Runtime |
| **pip** | Dependencies |
| **Docker Desktop** (optional) | Kafka broker for social trends pipeline |
| **API keys** (optional) | LLM (Groq/OpenAI/Gemini), X Developer account for social trends |

---

## Installation

```bash
# Clone the repository, then:
cd Real-Time-Stock-Analysis-using-Time-Series-and-Sentiment-Prediction

pip install -r requirements.txt
```

**Windows note:** If `streamlit` is not on PATH, use:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run.bat`.

**Important Google Trends dependency:**

```bash
pip install "urllib3>=1.26.18,<2" "pytrends>=4.9.2"
```

**Important Kafka client (social trends):**

```bash
pip install kafka-python
```

---

## Configuration

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` \| `groq` \| `gemini` |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | LLM provider key |
| `CHROMA_PERSIST_DIR` | RAG vector store (default `./data/chroma`) |
| `INFERENCE_DATA_DIR` | Transcripts folder (default `./data/transcripts`) |
| `SP500_CSV` | Ticker list (default `./data/datasets/SP500.csv`) |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | X OAuth1 for social trends |
| `X_AUTH_MODE` | `oauth1` (recommended) |
| `KAFKA_ENABLED` | `true` / `false` |
| `KAFKA_BOOTSTRAP_SERVERS` | Default `localhost:9092` |

Validate X credentials:

```bash
python scripts/validate_x_credentials.py
```

---

## Run the application

### 1. Streamlit UI (primary)

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

**Sidebar navigation:** Dashboard · Agentic AI · Overview · News · Company · Technicals · Search trends · Social trends · Meeting notes · Forecast

### 2. Index RAG corpus (first-time / after transcript updates)

```bash
python scripts/ingest_rag.py
```

Indexes `data/transcripts/` into Chroma at `data/chroma/`.

### 3. FastAPI backend (optional)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Docs: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health + RAG stats |
| `/chat` | POST | Agentic chat |
| `/analyze` | POST | Full ticker analysis |
| `/rag/query` | POST | Semantic transcript search |
| `/rag/ingest` | POST | Re-index corpus |
| `/tools` | GET | List agent tools |

### 4. MCP server (Cursor / Claude Desktop)

See `mcp_config.example.json`. Run:

```bash
python -m mcp_server.server
```

---

## Optional services

### Kafka (full X social pipeline)

```bash
# Windows
start_kafka.bat

# Or manually
docker compose -f infra/docker-compose.kafka.yml up -d
```

Install client library: `pip install kafka-python`

In the app: **Social trends** → sidebar **Reconnect Kafka**.

Standalone stream:

```bash
python scripts/run_x_kafka_stream.py --keyword Amazon --backfill
```

---

## Agent tools

| Tool | Data source |
|------|-------------|
| `get_stock_quote` | Yahoo Finance |
| `get_news_sentiment` | Finviz + VADER |
| `get_fundamentals` | Finviz |
| `get_technicals` | Price history indicators |
| `forecast_stock_price` | Prophet / trend model |
| `search_earnings_transcripts` | RAG / Chroma |
| `analyze_ticker` | Combined multi-source analysis |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Google Trends error on Search trends | `pip install "urllib3<2" pytrends`; wait 1 min if rate-limited; Finviz fallback loads automatically |
| Kafka not connected | `pip install kafka-python`; run `start_kafka.bat`; click **Reconnect Kafka** |
| X 402 Payment Required | X API credits required for tweet search — add credits at developer.x.com |
| X 401 Unauthorized | Regenerate access token/secret to match consumer key; set `X_AUTH_MODE=oauth1` |
| Prophet fails | App falls back to polynomial forecast automatically |
| RAG empty | Run `python scripts/ingest_rag.py` |

---

## Disclaimer

This platform provides automated financial analysis for **educational and research purposes only**. It is **not financial advice**. Forecasts, sentiment scores, and social signals are statistical estimates—not guarantees of future performance.
