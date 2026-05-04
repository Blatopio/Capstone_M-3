import streamlit as st
import pdfplumber
import io

from JobStation_app.graph.workflow import app
from JobStation_app.tools.utils import *

from langchain_core.messages import HumanMessage

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobStation",
    page_icon="💼",
    layout="centered"
)

# ─── SESSION STATE INIT ────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
if "username" not in st.session_state:
    st.session_state.username   = ""
if "role" not in st.session_state:
    st.session_state.role       = ""
if "messages" not in st.session_state:
    st.session_state.messages   = []


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def verify_login(username: str, password: str) -> dict | None:
    """Check credentials against MySQL users table. Returns user row or None."""
    try:
        conn   = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def extract_pdf_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF file."""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        ).strip()


def run_agent(user_input: str) -> str:
    """
    Invoke the LangGraph app with current session state.
    Returns the agent's final text response.
    """
    state = {
        "messages":  st.session_state.messages + [HumanMessage(content=user_input)],
        "next":      "",
        "role":      st.session_state.role,
        "username":  st.session_state.username,
    }

    result = app.invoke(state)

    # Extract last AI message
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai":
            if not getattr(msg, "tool_calls", None):
                return msg.content

    return "I could not generate a response. Please try again."


def get_all_candidates():
    """Fetch all candidates from MySQL for the admin panel."""
    conn   = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, category, prof_level, state, updated_at
        FROM candidates
        ORDER BY updated_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def update_candidate_state(candidate_id: int, new_state: str):
    """Update a candidate's state in MySQL."""
    conn   = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE candidates SET state = %s WHERE id = %s",
        (new_state, candidate_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


# ─── LOGIN SCREEN ──────────────────────────────────────────────────────────────
def show_login():
    st.title("💼 JobStation")
    st.caption("Direct placement. No salary deductions.")
    st.divider()

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.warning("Please enter both username and password.")
            return

        user = verify_login(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username  = user["username"]
            st.session_state.role      = user["role"]
            st.session_state.messages  = []
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ─── CHAT INTERFACE ────────────────────────────────────────────────────────────
def show_chat():
    # Display chat history
    for msg in st.session_state.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # CV upload for jobseekers
    if st.session_state.role == "jobseeker":
        with st.expander("📄 Upload your CV"):
            col1, col2 = st.columns([3, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload PDF", type=["pdf"], label_visibility="collapsed"
                )
            with col2:
                category = st.selectbox("Category", [
                    "HR", "ENGINEERING", "FINANCE", "INFORMATION-TECHNOLOGY",
                    "SALES", "MARKETING", "HEALTHCARE", "EDUCATION",
                    "BANKING", "CONSULTANT", "DESIGNER", "CHEF",
                    "ARTS", "AVIATION", "FITNESS", "ADVOCATE",
                    "ACCOUNTANT", "BUSINESS-DEVELOPMENT", "CONSTRUCTION",
                    "DIGITAL-MEDIA", "AGRICULTURE", "AUTOMOBILE",
                    "APPAREL", "BPO", "PUBLIC-RELATIONS", "TEACHER"
                ])

            if uploaded_file and st.button("Upload CV", use_container_width=True):
                with st.spinner("Processing your CV..."):
                    cv_text = extract_pdf_text(uploaded_file)
                    if cv_text:
                        prompt = (
                            f"Please upload my CV to the platform. "
                            f"My category is {category}. "
                            f"Here is my CV text:\n\n{cv_text[:3000]}"
                        )
                        response = run_agent(prompt)
                        st.session_state.messages.append(
                            HumanMessage(content=f"[CV Upload — {category}]")
                        )
                        from langchain_core.messages import AIMessage
                        st.session_state.messages.append(
                            AIMessage(content=response)
                        )
                        st.rerun()
                    else:
                        st.error("Could not extract text from PDF.")

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        from langchain_core.messages import AIMessage

        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_agent(prompt)
            st.markdown(response)

        st.session_state.messages.append(AIMessage(content=response))


# ─── ADMIN PANEL ───────────────────────────────────────────────────────────────
def show_admin():
    st.subheader("Candidate Management")
    st.caption("Update candidate states here. Changes are saved immediately.")

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_category = st.selectbox("Filter by category", ["All"] + [
            "HR", "ENGINEERING", "FINANCE", "INFORMATION-TECHNOLOGY",
            "SALES", "HEALTHCARE", "BANKING", "CONSULTANT"
        ])
    with col2:
        filter_level = st.selectbox("Filter by level",
            ["All", "junior", "senior", "specialist"])
    with col3:
        filter_state = st.selectbox("Filter by state",
            ["All", "available", "interviewed", "placed", "inactive"])

    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()

    candidates = get_all_candidates()

    # Apply filters
    if filter_category != "All":
        candidates = [c for c in candidates if c["category"] == filter_category]
    if filter_level != "All":
        candidates = [c for c in candidates if c["prof_level"] == filter_level]
    if filter_state != "All":
        candidates = [c for c in candidates if c["state"] == filter_state]

    st.caption(f"Showing {len(candidates)} candidates")
    st.divider()

    if not candidates:
        st.info("No candidates match the selected filters.")
        return

    # Candidate rows
    STATE_OPTIONS = ["available", "interviewed", "placed", "inactive"]

    for candidate in candidates:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

        with col1:
            st.caption("ID")
            st.write(str(candidate["id"]))
        with col2:
            st.caption("Category / Level")
            st.write(f"{candidate['category']} — {candidate['prof_level']}")
        with col3:
            st.caption("Current state")
            state_color = {
                "available":   "🟢",
                "interviewed": "🟡",
                "placed":      "🔵",
                "inactive":    "⚫"
            }
            st.write(f"{state_color.get(candidate['state'], '')} {candidate['state']}")
        with col4:
            st.caption("Update state")
            new_state = st.selectbox(
                label="state",
                options=STATE_OPTIONS,
                index=STATE_OPTIONS.index(candidate["state"]),
                key=f"state_{candidate['id']}",
                label_visibility="collapsed"
            )
            if new_state != candidate["state"]:
                update_candidate_state(candidate["id"], new_state)
                st.success("Updated!")
                st.rerun()

        st.divider()


# ─── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    # Sidebar
    with st.sidebar:
        st.title("💼 JobStation")
        st.caption(f"Logged in as **{st.session_state.username}**")
        st.caption(f"Role: `{st.session_state.role}`")
        st.divider()

        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Role-based tabs
    if st.session_state.role == "company":
        chat_tab, admin_tab = st.tabs(["💬 Chat", "🗂️ Candidates"])
        with chat_tab:
            show_chat()
        with admin_tab:
            show_admin()
    else:
        st.header(f"Welcome, {st.session_state.username}! 👋")
        show_chat()


if __name__ == "__main__":
    main()