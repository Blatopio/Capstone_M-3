# JobStation 💼

> **Direct placement. No salary deductions.**  
> An AI-powered job placement platform where jobseekers upload their CV once and get matched to the right opportunities — built with LangGraph, Qdrant, and Streamlit.

🌐 **Live App:** [jobstationapp.streamlit.app](https://jobstationapp.streamlit.app/)

---

## What is JobStation?

JobStation connects candidates directly with companies — no outsourcing middlemen, no salary deductions. Jobseekers upload their CV once, and the AI handles the rest: extracting skills, inferring experience level, embedding the content into a vector database, and surfacing the best matching job categories.

The platform is built around a **multi-agent AI system** using LangGraph, where a supervisor routes each conversation to the right specialized agent based on the user's role and intent.

---

## Current Status

| Role | Status | Notes |
|---|---|---|
| 👤 Jobseeker | ✅ Fully working | CV upload, recommendations, career chat |
| 🏢 Company / HR | 🚧 Under development | Candidate search planned for future release |

> **Note for HR accounts:** The company features (candidate search and admin panel) are currently under active development and will be available in a future update. The platform is fully functional for jobseekers.

---

## Features (Jobseeker)

- **One-time CV upload** — Upload your CV as a PDF once. The system extracts the text, embeds it into Qdrant, and remembers it across all future sessions. No re-upload needed.
- **Automatic experience level detection** — The system reads your CV and infers your proficiency level (`junior`, `senior`, or `specialist`) automatically.
- **Job category recommendations** — Ask the agent what jobs fit your background and it searches the vector database for the best matching categories.
- **Persistent CV memory** — Log out, come back tomorrow, and the agent still knows your background.
- **Career assistant** — Ask anything: CV tips, interview advice, career path suggestions, platform questions, or general knowledge.
- **Chat history** — Full conversation context maintained across turns.
- **Usage transparency** — Input and output token counts shown per message.

---

## How It Works

```
Jobseeker logs in
        │
        ▼
  Supervisor agent
  (reads intent)
        │
        ├── "upload my CV" ──► Jobseeker agent
        │                           │
        │                     upload_cv_tool
        │                           │
        │                  Embed text → Qdrant
        │                  Save qdrant_id → MySQL
        │
        ├── "what jobs fit me?" ──► Jobseeker agent
        │                               │
        │                    get_recommendations_tool
        │                               │
        │                  Fetch CV from Qdrant by username
        │                  Semantic search → top categories
        │
        └── anything else ──► General agent
                                    │
                          CV context injected from Qdrant
                          Answers career/platform questions
                                    │
                                   END
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph + LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector database | Qdrant Cloud |
| Relational database | MySQL (Railway) |
| Observability | Langfuse |
| PDF parsing | pdfplumber |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
Capstone_M-3/
├── main.py                        # Streamlit entry point
├── requirements.txt               # Python dependencies
├── README.md
└── JobStation_app/
    ├── config.py                  # LLM, embeddings, Langfuse, Qdrant setup
    ├── graph/
    │   ├── agents.py              # Supervisor, jobseeker, general, company agents
    │   ├── state.py               # LangGraph state schema
    │   └── workflow.py            # Graph nodes, edges, routing logic
    └── tools/
        ├── tools.py               # upload_cv_tool, get_recommendations_tool
        └── utils.py               # MySQL + Qdrant helpers, CV fetch logic
```

---

## Setup (Local)

### 1. Clone the repo
```bash
git clone https://github.com/Blatopio/Capstone_M-3.git
cd Capstone_M-3
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```env
OPENAI_API_KEY=sk-...

QDRANT_URL=https://...
QDRANT_API_KEY=...

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=...
MYSQL_DATABASE=jobstationdatabase

LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Run
```bash
streamlit run main.py
```

---

## Demo Accounts

| Username | Password | Role |
|---|---|---|
| jobseeker1 | `****` | Jobseeker ✅ |
| company1 | `****` | Company 🚧 |

---

## Usage Guide

### Uploading your CV
1. Log in as `jobseeker1`
2. Click **"📄 Upload your CV"** expander
3. Select your PDF and choose your job category
4. Click **"Upload CV"**
5. The agent confirms upload and your CV is now stored

### Getting job recommendations
After uploading, simply type in the chat:
> *"What job recommendations do you have based on my CV?"*

The agent fetches your stored CV from Qdrant automatically and returns the best matching categories — no need to upload again.

### Career questions
You can ask anything:
> *"What are my strongest skills based on my CV?"*  
> *"How should I prepare for an engineering interview?"*  
> *"What companies should I target with my background?"*

---

## Observability

All LLM calls are traced in **Langfuse** with session and user tagging. Each chat message in the UI shows three collapsible panels:

- **🔧 Tool Calls** — raw output when RAG tools are invoked (proves answers come from the vector DB, not hallucination)
- **🕘 History Chat** — full conversation context across turns
- **📊 Usage Details** — input and output token counts

---

## Future Development

- 🚧 **Company candidate search** — semantic search by job description, category, and experience level
- 🚧 **Admin panel** — HR updates candidate state (available → interviewed → placed → inactive)
- 🚧 **Multi-CV support** — jobseekers can manage multiple CV versions
- 🚧 **Notification system** — alert jobseekers when their profile is viewed
- 🚧 **Analytics dashboard** — placement rates, category trends, time-to-placement

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

---

## License

Built for educational purposes as part of the Purwadhika bootcamp curriculum.
