# AI Research Helper

An AI-powered research assistant that helps you search academic papers on arXiv, summarize them, manage citations, and chat with an assistant that can pull in relevant papers automatically. Built with LangChain, Groq (LLM), and Streamlit.

## Features

- 💬 **Chat interface** — ask research questions in natural language; the assistant decides whether to search arXiv or answer directly
- 🔍 **Paper search** — search arXiv for academic papers, displayed as result cards with title, authors, date, and abstract
- 📝 **Summarization** — one-click AI summary of any paper's abstract
- 📌 **Citation management** — save citations from search results and export them as a `.txt` file
- 🕘 **Search history** — past searches are logged with timestamps (SQLite)
- ✅ **Input validation** — empty/too-short/too-long queries are rejected with a clear message before hitting any API

## Tech stack

- LangChain + Groq (`llama-3.3-70b-versatile`)
- `arxiv` Python package (free, no API key required)
- SQLite (search history)
- Streamlit (UI + deployment)

## Setup — run locally

1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.streamlit/secrets.toml` file in the project root:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
   Get a free key at [console.groq.com](https://console.groq.com).
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Setup — deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, and set the main file to `app.py`.
4. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Deploy. First build takes a couple of minutes.

## Project structure

```
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Known limitations

- Search history and citations are stored per-session / in a local SQLite file, which resets when the Streamlit Cloud app reboots or redeploys. For persistent storage across restarts, a hosted database (e.g. Supabase, PostgreSQL) would be needed.
- Web/paper search quality depends on the arXiv API and the underlying LLM's routing decisions; results may occasionally be imperfect for very broad or ambiguous queries.
