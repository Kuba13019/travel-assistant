import os
import sqlite3
from datetime import datetime

import streamlit as st
import arxiv
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Research Helper", page_icon="📚", layout="wide")
st.title("📚 AI Research Helper")
st.caption("arXiv Search + Summarization + Citations + History + Chat — powered by LangChain + Groq")

# ---------- API key ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY set nahi hai. Streamlit Cloud app settings -> Secrets me add karo.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)

# ---------- SQLite setup (search history) ----------
DB_PATH = "research_helper.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_search(query: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO search_history (query, timestamp) VALUES (?, ?)",
        (query, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT query, timestamp FROM search_history ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows


init_db()

# ---------- Session state ----------
if "papers" not in st.session_state:
    st.session_state.papers = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ---------- Input validation ----------
def validate_query(query: str):
    """Returns (is_valid, error_message)."""
    query = query.strip()
    if not query:
        return False, "Search query khali nahi ho sakti."
    if len(query) < 3:
        return False, "Search query kam se kam 3 characters ki honi chahiye."
    if len(query) > 300:
        return False, "Search query bahut lambi hai (max 300 characters)."
    return True, ""


# ---------- Helper functions ----------
def search_arxiv(query: str, max_results: int = 5):
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published.date()),
                "url": paper.entry_id,
            })
        return results
    except Exception as e:
        st.error(f"arXiv search error: {e}")
        return []


def summarize_text(text: str) -> str:
    try:
        prompt = f"Summarize this research paper abstract in 3-4 simple sentences:\n\n{text}"
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Summarization error: {e}"


def format_citation(paper: dict) -> str:
    authors = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors += " et al."
    year = paper["published"][:4]
    return f"{authors} ({year}). {paper['title']}. arXiv. {paper['url']}"


def chat_with_assistant(user_input: str) -> str:
    """Simple router: decide if the question needs an arXiv search or can be answered directly."""
    router_prompt = f"""You are a research assistant. Decide if answering this question requires
searching arXiv for papers, or if it can be answered directly from general knowledge.

Respond with ONLY a JSON object: {{"needs_search": true/false, "search_query": "..."}}

User question: {user_input}"""
    try:
        import json
        import re
        route_raw = llm.invoke(router_prompt).content
        match = re.search(r"\{.*\}", route_raw, re.DOTALL)
        needs_search = False
        search_query = user_input
        if match:
            parsed = json.loads(match.group(0))
            needs_search = parsed.get("needs_search", False)
            search_query = parsed.get("search_query", user_input)
    except Exception:
        needs_search = False
        search_query = user_input

    if needs_search:
        papers = search_arxiv(search_query, max_results=3)
        if not papers:
            context = "No relevant papers found on arXiv."
        else:
            context = "\n\n".join(
                f"- {p['title']} ({p['published'][:4]}): {p['summary'][:200]}..." for p in papers
            )
        final_prompt = (
            f"User question: {user_input}\n\nRelevant papers found:\n{context}\n\n"
            f"Answer the user's question using these papers, in a helpful and concise way."
        )
    else:
        final_prompt = f"Answer this research-related question helpfully and concisely: {user_input}"

    try:
        return llm.invoke(final_prompt).content
    except Exception as e:
        return f"Sorry, kuch galat ho gaya: {e}"


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Search Papers", "📌 Citations", "🕘 Search History"])

with tab1:
    st.subheader("Ask the Research Assistant")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    chat_input = st.chat_input("Ask a research question...")
    if chat_input:
        is_valid, error_msg = validate_query(chat_input)
        if not is_valid:
            st.warning(error_msg)
        else:
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.write(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = chat_with_assistant(chat_input)
                st.write(reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})

with tab2:
    st.subheader("Search academic papers on arXiv")
    query = st.text_input("Search query", placeholder="e.g. large language models reasoning")

    if st.button("Search"):
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            st.warning(error_msg)
        else:
            with st.spinner("Searching arXiv..."):
                st.session_state.papers = search_arxiv(query)
                save_search(query)

    for i, paper in enumerate(st.session_state.papers):
        with st.container(border=True):
            st.markdown(f"**{paper['title']}**")
            st.caption(f"{', '.join(paper['authors'][:3])} — {paper['published']}")
            st.write(paper["summary"][:300] + "...")
            st.link_button("View on arXiv", paper["url"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Summarize", key=f"summarize_{i}"):
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(paper["summary"])
                    st.info(summary)
            with col2:
                if st.button("Add to Citations", key=f"cite_{i}"):
                    citation = format_citation(paper)
                    if citation not in st.session_state.citations:
                        st.session_state.citations.append(citation)
                        st.success("Added to citations!")

with tab3:
    st.subheader("Your saved citations")
    if not st.session_state.citations:
        st.write("Koi citation abhi tak add nahi hui. Search tab se papers add karo.")
    else:
        for c in st.session_state.citations:
            st.write(f"- {c}")

        citation_text = "\n".join(st.session_state.citations)
        st.download_button(
            "Download citations (.txt)",
            citation_text,
            file_name="citations.txt",
        )

        if st.button("Clear all citations"):
            st.session_state.citations = []
            st.rerun()

with tab4:
    st.subheader("Recent searches")
    history = get_history()
    if not history:
        st.write("Koi search history nahi hai abhi tak.")
    else:
        for q, ts in history:
            st.write(f"- **{q}** — {ts}")
    st.caption(
        "Note: Streamlit Cloud pe storage temporary hai — reboot/redeploy hone par history reset ho sakti hai."
    )
