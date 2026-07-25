import os
import sqlite3
from datetime import datetime

import streamlit as st
import arxiv
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Research Helper", page_icon="📚")
st.title("📚 AI Research Helper")
st.caption("arXiv Search + Summarization + Citations + History — powered by LangChain + Groq")

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


# ---------- Helper functions ----------
def search_arxiv(query: str, max_results: int = 5):
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in search.results():
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


# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Search Papers", "📌 Citations", "🕘 Search History"])

with tab1:
    st.subheader("Search academic papers on arXiv")
    query = st.text_input("Search query", placeholder="e.g. large language models reasoning")

    if st.button("Search") and query:
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

with tab2:
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

with tab3:
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
