# JobStation 💼

> **Direct placement. No salary deductions.**  
> A multi-agent AI-powered job placement platform connecting companies directly with candidates — built with LangGraph, Qdrant, and Streamlit.

---

## Overview

JobStation is a Capstone Project for Module 3, demonstrating end-to-end implementation of a RAG-based multi-agent system. The platform serves two roles:

- **Jobseekers** — upload their CV once, get job category recommendations, and become discoverable by companies
- **Companies (HR)** — search for candidates semantically by job category and experience level

The system uses a **supervisor agent architecture** built on LangGraph, with Qdrant as the vector database for semantic CV search, MySQL for relational data, and Langfuse for full LLM observability.

---

## Features

- **Multi-agent workflow** — Supervisor routes each message to the correct specialized agent (jobseeker, company, or general)
- **RAG-powered CV search** — CV text is embedded and stored in Qdrant; companies search by natural language description
- **Persistent CV memory** — Jobseekers upload once; the agent fetches their CV automatically in every session
- **Role-based access** — Separate chat experience for jobseekers and company HR users
- **Admin panel** — Companies can update candidate states (available → interviewed → placed → inactive)
- **Chat history** — Full conversation context maintained across turns
- **LLM observability** — Every agent call, tool invocation, and token usage tracked via Langfuse
- **Usage transparency** — Input/output token counts shown per message in the UI

---

## Architecture

```
User Login
    │
    ▼
Streamlit UI  ──────────────────────────────────────────
    │                                                   │
    ▼                                               Admin panel
Supervisor (LangGraph)                            (company only)
    │
    ├─── Jobseeker agent ──► upload_cv_tool ──► Qdrant + MySQL
    │                    └──► get_recommendations_tool ──► Qdrant
    │
    ├─── General agent ──► Answers directly (CV context injected)
    │                  └──► END (no loop)
    │
    └─── Company agent ──► search_candidates_tool
                        └──► SQL agent + Qdrant semantic search

All agents observed by Langfuse
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph + LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Qdrant Cloud |
| Relational DB | MySQL |
| Observability | Langfuse |
| PDF parsing | pdfplumber |

---

## Project Structure

```
JobStation_app/
├── graph/
│   ├── agents.py       # Supervisor, jobseeker, company, general agents
│   ├── state.py        # LangGraph state schema
│   └── workflow.py     # Graph nodes, edges, and compilation
├── tools/
│   ├── tools.py        # upload_cv_tool, get_recommendations_tool, search_candidates_tool
│   └── utils.py        # MySQL + Qdrant helpers, CV fetch logic
├── config.py           # LLM, embeddings, Langfuse, Qdrant client setup
└── __init__.py
main.py                 # Streamlit app entry point
requirements.txt
.env                    # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Blatopio/jobstation.git
cd jobstation
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant Cloud
QDRANT_URL=https://...
QDRANT_API_KEY=...

# MySQL
MYSQL_HOST=...
MYSQL_PORT=3306
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DATABASE=...

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Run the app

```bash
streamlit run main.py
```

---

## Database Schema

### `users` table

| Column | Type | Description |
|---|---|---|
| username | VARCHAR | Primary key, login identifier |
| password | VARCHAR | User password |
| role | ENUM | `jobseeker` or `company` |
| qdrant_id | VARCHAR | UUID of the user's CV point in Qdrant |

### `candidates` table

| Column | Type | Description |
|---|---|---|
| id | INT | Primary key |
| category | VARCHAR | Job category (e.g. ENGINEERING, HR) |
| prof_level | ENUM | `junior`, `senior`, or `specialist` |
| state | ENUM | `available`, `interviewed`, `placed`, `inactive` |
| qdrant_id | VARCHAR | UUID reference to Qdrant vector point |

---

## How It Works

### Jobseeker flow

1. Log in as a jobseeker
2. Upload CV as PDF — text is extracted, embedded, and stored in Qdrant; `qdrant_id` saved to MySQL
3. In future sessions, ask *"what job recommendation for me?"* — the agent fetches the CV automatically from Qdrant using the stored `qdrant_id`, no re-upload needed
4. Companies can now find the profile via semantic search

### Company flow

1. Log in as a company HR user
2. Chat to search: *"find me a senior engineer with automation experience"*
3. The agent calls `search_candidates_tool` → SQL agent filters by category and level → Qdrant semantic search ranks by similarity
4. View and manage candidate states in the Admin panel tab

### Agent routing

The **Supervisor** reads each message and routes to:
- `jobseeker_agent` — when a jobseeker wants to upload or get recommendations
- `company_agent` — when HR wants to search candidates
- `general_agent` — for greetings, platform questions, career advice, and general knowledge

The `general_agent` terminates directly at `END` to prevent infinite loops.

---

## Observability

All LLM calls, tool invocations, and token usage are traced in **Langfuse**. Each session is tagged with the logged-in username. The Streamlit UI also shows per-message:

- **Tool Calls** — raw tool output (visible when RAG tools are invoked)
- **History Chat** — full conversation context
- **Usage Details** — input and output token counts

---

## Proficiency Level Inference

When a CV is uploaded, the system automatically infers the candidate's proficiency level from the CV text:

| Level | Criteria |
|---|---|
| `junior` | ≤ 2 years of experience (or not detectable) |
| `senior` | ≤ 3 years of experience |
| `specialist` | > 3 years of experience |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.

Live URL: `https://Blatopio-jobstation-main.streamlit.app`

Secrets are managed via Streamlit Cloud's secrets manager (equivalent to `.env`).

---

## Assessment Components

| Component | Weight |
|---|---|
| Video explanation | 25% |
| Vector database implementation | 10% |
| RAG tool implementation | 10% |
| Agent implementation | 20% |
| Prompt engineering | 10% |
| Streamlit integration | 25% |

---

## Author

**Muhammad Fachreza Alghifari**  
Purwadhika Digital Technology School — Module 3 Capstone Project

---

## License

This project is built for educational purposes as part of the Purwadhika bootcamp curriculum.
