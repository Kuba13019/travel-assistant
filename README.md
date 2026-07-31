# AI Travel Assistant

An AI-powered travel assistant with a RAG chatbot, live web search, weather lookup, flight search, and AI-generated itineraries. Built with LangChain, Groq (LLM), and Streamlit.

## Features

- 💬 **Chat interface** — ask travel questions; the assistant routes to Web Search, Weather, or your uploaded travel document as needed
- 📄 **Document upload (RAG)** — upload a travel guide (PDF/TXT) and ask questions about it
- 🌦️ **Weather lookup** — current weather for any city (OpenWeatherMap)
- ✈️ **Flight search** — search flights between airports (Aviationstack API)
- 🗺️ **Itinerary generator** — AI-generated day-by-day itinerary for any destination
- 🕘 **Search history** — flight and itinerary searches are logged (SQLite)
- ✅ **Input validation** — empty/invalid inputs are rejected with a clear message

## Tech stack

- LangChain + Groq (`llama-3.3-70b-versatile`)
- FAISS (vector store for RAG)
- DuckDuckGo Search (web search tool)
- OpenWeatherMap API (weather)
- Aviationstack API (flights)
- SQLite (search history)
- Streamlit (UI + deployment)

## Setup — run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   OPENWEATHER_API_KEY = "your-openweather-api-key"
   AVIATIONSTACK_API_KEY = "your-aviationstack-api-key"
   ```
3. Run:
   ```bash
   streamlit run app.py
   ```

## Setup — deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → select this repo → main file `app.py`.
4. In **Settings → Secrets**, add all three keys shown above.
5. Deploy.

## Known limitations

- Search history (SQLite) resets when the Streamlit Cloud app reboots/redeploys — a hosted database would be needed for persistence.
- Aviationstack free tier has limited monthly requests and mainly returns real-time/scheduled flight data rather than bookable fares.
- Web search quality depends on DuckDuckGo's free API, which occasionally returns no results for very broad queries.
