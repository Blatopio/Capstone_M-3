import uuid
import re
from collections import Counter

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from qdrant_client.models import Filter, FieldCondition, MatchAny

from JobStation_app.tools.utils import *
from JobStation_app.config import llm, langfuse_handler, QDRANT_URL, QDRANT_API_KEY


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _infer_prof_level(cv_text: str) -> str:
    """
    Heuristic: scan CV text for years-of-experience clues.
    Returns 'junior' (≤2 yrs), 'senior' (≤3 yrs), or 'specialist' (>3 yrs).
    """
    # Look for patterns like "5 years", "3+ years", "2 tahun", etc.
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|tahun)\s*(?:of)?\s*(?:experience|pengalaman)",
        r"(?:experience|pengalaman)[^\d]*(\d+)\s*\+?\s*(?:years?|tahun)",
    ]
    years_found = []
    for pat in patterns:
        for match in re.finditer(pat, cv_text, re.IGNORECASE):
            years_found.append(int(match.group(1)))

    if not years_found:
        return "junior"        # default when we cannot detect

    max_years = max(years_found)
    if max_years <= 2:
        return "junior"
    elif max_years <= 3:
        return "senior"
    else:
        return "specialist"


# ─── SQL AGENT ────────────────────────────────────────────────────────────────
def get_sql_agent():
    """
    Returns a read-only SQL agent connected to the candidates table.
    Uses mysql.connector as the driver via SQLAlchemy URI.
    System prompt enforces state='available' and SELECT-only rules.
    """
    db = SQLDatabase.from_uri(
        MYSQL_URI,
        include_tables=["candidates"],
        sample_rows_in_table_info=2,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        prefix="""
You are a SQL expert helping to find job candidates from a database.

CRITICAL RULES — always follow these without exception:
1. ONLY run SELECT statements. Never UPDATE, DELETE, INSERT, or DROP.
2. ALWAYS include state = 'available' in every WHERE clause.
3. Only query the 'candidates' table.
4. Return candidate IDs and their details clearly.

The candidates table has these columns:
- id          : candidate ID (INT)
- category    : job category e.g. HR, ENGINEERING, FINANCE (VARCHAR uppercase)
- prof_level  : junior, senior, or specialist (ENUM)
- state       : available, interviewed, placed, inactive (ENUM)
- qdrant_id   : reference to vector in Qdrant (VARCHAR)
        """,
    )
    return agent


# ─── TOOLS ────────────────────────────────────────────────────────────────────
@tool
def search_candidates_tool(query: str, category: str, prof_level: str, top_k: int = 3) -> str:
    """
    Search for matching candidates for a company/HR query.

    Step 1 — SQL Agent: generates and runs SELECT query filtered by
             category, prof_level, state='available'
    Step 2 — Qdrant: semantic search within that filtered pool

    Args:
        query:      what the company is looking for
                    e.g. "team leader with conflict resolution experience"
        category:   job category e.g. "HR", "ENGINEERING", "FINANCE"
        prof_level: "junior", "senior", or "specialist"
        top_k:      number of candidates to return (default 3)
    """
    sql_agent  = get_sql_agent()
    sql_prompt = (
        f"Find all available candidates where category is '{category.upper()}' "
        f"and prof_level is '{prof_level.lower()}' and state is 'available'. "
        f"Return their id and qdrant_id."
    )

    try:
        sql_result = sql_agent.invoke(
            {"input": sql_prompt},
            config={"callbacks": [langfuse_handler]},   # ← Langfuse trace
        )
        sql_output = sql_result.get("output", "")
    except Exception as e:
        return f"SQL query failed: {str(e)}"

    # Step 2 — Qdrant semantic search within filtered pool
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.category",
                match=MatchAny(any=[category.upper()])
            ),
            FieldCondition(
                key="metadata.prof_level",
                match=MatchAny(any=[prof_level.lower()])
            ),
        ]
    )

    vectorstore = get_qdrant_vectorstore()
    results = vectorstore.similarity_search(
        query=query,
        k=top_k,
        filter=qdrant_filter,
    )

    if not results:
        return "No semantically matching candidates found for your query."

    output = f"Top {len(results)} matching candidates for '{query}':\n\n"
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        output += f"Candidate {i}:\n"
        output += f"  ID       : {meta.get('candidate_id', 'N/A')}\n"
        output += f"  Category : {meta.get('category', 'N/A')}\n"
        output += f"  Level    : {meta.get('prof_level', 'N/A')}\n"
        output += f"  Preview  : {doc.page_content[:300]}...\n\n"

    return output


@tool
def upload_cv_tool(cv_text: str, category: str, username: str) -> str:
    """
    Upload a jobseeker's CV to the platform.

    Step 1 — Infer prof_level from CV text (junior / senior / specialist)
    Step 2 — Generate UUID as the new Qdrant point ID
    Step 3 — Embed CV text and upsert into Qdrant
    Step 4 — Insert new row into MySQL candidates table
    Step 5 — Update users.qdrant_id for this jobseeker

    Args:
        cv_text:  full text of the jobseeker's CV
        category: job category e.g. "HR", "ENGINEERING"
        username: the logged-in jobseeker's username
    """
    # ── FIX: infer prof_level from CV rather than always using 'junior' ───────
    prof_level    = _infer_prof_level(cv_text)
    new_qdrant_id = str(uuid.uuid4())

    doc = Document(
        page_content=cv_text,
        metadata={
            "candidate_id": new_qdrant_id,
            "category":     category.upper(),
            "prof_level":   prof_level,         # ← was always "junior" before
            "source":       "jobseeker_upload",
        }
    )
    vectorstore = get_qdrant_vectorstore()
    vectorstore.add_documents([doc])

    conn   = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO candidates (id, category, prof_level, state, qdrant_id)
        VALUES (%s, %s, %s, 'available', %s)
    """, (abs(hash(new_qdrant_id)) % (10**9), category.upper(), prof_level, new_qdrant_id))

    cursor.execute("""
        UPDATE users
        SET qdrant_id = %s
        WHERE username = %s
    """, (new_qdrant_id, username))

    conn.commit()
    cursor.close()
    conn.close()

    return (
        f"✅ CV uploaded successfully!\n"
        f"   Category     : {category.upper()}\n"
        f"   Level        : {prof_level}\n"
        f"   Your CV is now visible to companies.\n"
        f"   Reference ID : {new_qdrant_id}"
    )


@tool
def get_recommendations_tool(cv_text: str, top_k: int = 3) -> str:
    """
    Given a jobseeker's CV text, find similar profiles in the database
    and recommend the best matching job categories.

    Args:
        cv_text: the jobseeker's CV text
        top_k:   number of similar profiles to retrieve (default 3)
    """
    vectorstore = get_qdrant_vectorstore()
    results     = vectorstore.similarity_search(cv_text, k=top_k)

    if not results:
        return "No similar profiles found in our database."

    categories    = [doc.metadata.get("category", "UNKNOWN") for doc in results]
    top_categories = Counter(categories).most_common()

    output = "Based on your CV, here are the best matching job categories:\n\n"
    for cat, count in top_categories:
        output += f"  • {cat} ({count} similar profile(s) found)\n"

    output += "\nSimilar profiles in our database:\n\n"
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        output += f"Profile {i}:\n"
        output += f"  Category : {meta.get('category', 'N/A')}\n"
        output += f"  Level    : {meta.get('prof_level', 'N/A')}\n"
        output += f"  Preview  : {doc.page_content[:200]}...\n\n"

    return output