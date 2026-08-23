import os
import json
import re
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️", layout="wide")
st.title("✈️ AI Travel Assistant")
st.caption("RAG Chat + Web Search + Weather + Flights + Itinerary — powered by LangChain + Groq")

# ---------- API keys (Streamlit Cloud secrets, or local env vars) ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.environ.get("OPENWEATHER_API_KEY", ""))
AVIATIONSTACK_API_KEY = st.secrets.get("AVIATIONSTACK_API_KEY", os.environ.get("AVIATIONSTACK_API_KEY", ""))

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY set nahi hai. Streamlit Cloud app settings -> Secrets me add karo.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)


@st.cache_resource
def load_embeddings():
    return FakeEmbeddings(size=768)


embeddings_model = load_embeddings()

# ---------- SQLite setup (saved searches) ----------
DB_PATH = "travel_assistant.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_type TEXT,
            details TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_search(search_type: str, details: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO searches (search_type, details, timestamp) VALUES (?, ?, ?)",
        (search_type, details, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_searches():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT search_type, details, timestamp FROM searches ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return rows


init_db()

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_itinerary" not in st.session_state:
    st.session_state.last_itinerary = None
if "last_itinerary_dest" not in st.session_state:
    st.session_state.last_itinerary_dest = ""
if "last_flights" not in st.session_state:
    st.session_state.last_flights = None
if "last_weather" not in st.session_state:
    st.session_state.last_weather = None


# ---------- Input validation ----------
def validate_text_input(value: str, field_name: str, min_len: int = 2, max_len: int = 200):
    value = value.strip()
    if not value:
        return False, f"{field_name} khali nahi ho sakta."
    if len(value) < min_len:
        return False, f"{field_name} kam se kam {min_len} characters ka hona chahiye."
    if len(value) > max_len:
        return False, f"{field_name} bahut lamba hai (max {max_len} characters)."
    return True, ""


def validate_iata_code(value: str, field_name: str):
    value = value.strip().upper()
    if not value:
        return False, f"{field_name} khali nahi ho sakta."
    if not re.fullmatch(r"[A-Z]{3}", value):
        return False, f"{field_name} exactly 3 letters ka IATA code hona chahiye (e.g. DEL, BOM)."
    return True, ""


# ---------- Sidebar: document upload (RAG) ----------
st.sidebar.header("📄 Travel Document Upload")
uploaded_file = st.sidebar.file_uploader("PDF ya TXT travel guide upload karo", type=["pdf", "txt"])

if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings_model)
        st.sidebar.success(f"Document processed! ({len(chunks)} chunks)")
    except Exception as e:
        st.sidebar.error(f"Document process karne me error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------- Tool functions ----------
search_tool_runner = DuckDuckGoSearchRun()


def run_web_search(query: str) -> str:
    try:
        result = search_tool_runner.run(query)
        if "No good DuckDuckGo Search Result" in result or not result.strip():
            simplified = re.sub(
                r"\b(right now|currently|today|at the moment|these days)\b", "", query, flags=re.IGNORECASE
            ).strip()
            if simplified and simplified != query:
                result = search_tool_runner.run(simplified)
        return result
    except Exception as e:
        return f"Web search error: {e}"


def run_weather_lookup(city: str):
    """Returns (result_dict_or_None, error_message_or_None) for card display, plus a text summary."""
    if not OPENWEATHER_API_KEY:
        return None, "Weather API key configured nahi hai."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("cod") != 200:
            return None, f"Sorry, '{city}' ka weather nahi mil paya. ({data.get('message', 'unknown error')})"
        result = {
            "city": data.get("name", city),
            "country": data.get("sys", {}).get("country", ""),
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind": data.get("wind", {}).get("speed", 0),
            "description": data["weather"][0]["description"].title(),
        }
        return result, None
    except Exception as e:
        return None, f"Weather fetch error: {e}"


def run_weather_lookup_text(city: str) -> str:
    """Text-only version used by the chat router tool."""
    result, error = run_weather_lookup(city)
    if error:
        return error
    return f"{result['city']} mein abhi {result['temp']}°C hai, aasman: {result['description']}"


def run_doc_qa(query: str) -> str:
    if st.session_state.vectorstore is None:
        return "Koi document upload nahi kiya gaya hai."
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain.run(query)


def search_flights(dep_iata: str, arr_iata: str):
    if not AVIATIONSTACK_API_KEY:
        return None, "Aviationstack API key configured nahi hai."
    url = (
        f"http://api.aviationstack.com/v1/flights"
        f"?access_key={AVIATIONSTACK_API_KEY}&dep_iata={dep_iata}&arr_iata={arr_iata}&limit=5"
    )
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        if "error" in data:
            return None, data["error"].get("message", "Unknown API error")
        flights = data.get("data", [])
        if not flights:
            return [], None
        results = []
        for f in flights:
            results.append({
                "airline": f.get("airline", {}).get("name", "N/A"),
                "flight_number": f.get("flight", {}).get("iata", "N/A"),
                "departure_airport": f.get("departure", {}).get("airport", "N/A"),
                "departure_time": f.get("departure", {}).get("scheduled", "N/A"),
                "arrival_airport": f.get("arrival", {}).get("airport", "N/A"),
                "arrival_time": f.get("arrival", {}).get("scheduled", "N/A"),
                "status": f.get("flight_status", "N/A"),
            })
        return results, None
    except Exception as e:
        return None, str(e)


def generate_itinerary(destination: str, days: int, interests: str) -> str:
    prompt = (
        f"Create a simple day-by-day travel itinerary for {days} days in {destination}. "
        f"The traveler is interested in: {interests if interests else 'general sightseeing'}. "
        f"Format it clearly with a '## Day N: <short title>' heading for each day, followed by "
        f"2-3 bullet point activities. Keep it concise."
    )
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Itinerary generation error: {e}"


# ---------- Chat router ----------
def route_and_answer(user_input: str) -> str:
    doc_available = st.session_state.vectorstore is not None
    tool_list_text = (
        '- "WebSearch": current events, prices, live/recent travel info\n'
        '- "WeatherLookup": current weather for a city (input = just the city name)\n'
    )
    if doc_available:
        tool_list_text += '- "TravelDocsQA": answer using the uploaded travel guide document\n'
    tool_list_text += '- "none": answer directly from your own knowledge, no tool needed\n'

    router_prompt = f"""You are a routing assistant. Given the user's question, decide which ONE tool to use.

Available tools:
{tool_list_text}
Respond with ONLY a JSON object, nothing else, in this exact format:
{{"tool": "ToolName", "input": "text to pass to the tool"}}

User question: {user_input}"""

    try:
        route_response = llm.invoke(router_prompt).content
    except Exception as e:
        return f"Sorry, routing failed: {e}"

    match = re.search(r"\{.*\}", route_response, re.DOTALL)
    tool_name, tool_input = "none", user_input
    if match:
        try:
            parsed = json.loads(match.group(0))
            tool_name = parsed.get("tool", "none")
            tool_input = parsed.get("input", user_input)
        except Exception:
            pass

    if tool_name == "WebSearch":
        tool_result = run_web_search(tool_input)
    elif tool_name == "WeatherLookup":
        tool_result = run_weather_lookup_text(tool_input)
    elif tool_name == "TravelDocsQA" and doc_available:
        tool_result = run_doc_qa(tool_input)
    else:
        tool_result = None

    if tool_result is None:
        final_prompt = f"Answer this travel-related question helpfully and concisely: {user_input}"
    else:
        final_prompt = (
            f"User question: {user_input}\n\nTool used: {tool_name}\nTool result: {tool_result}\n\n"
            f"Using the tool result above, give a natural, helpful, concise answer to the user's question."
        )

    try:
        return llm.invoke(final_prompt).content
    except Exception as e:
        return f"Sorry, kuch galat ho gaya: {e}"


# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Chat", "🌦️ Weather", "✈️ Flights", "🗺️ Itinerary", "🕘 Search History"]
)

# ===================== TAB 1: CHAT =====================
with tab1:
    st.subheader("Chat with your Travel Assistant")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Apni trip ke baare me kuch bhi pucho...")
    if user_input:
        is_valid, error_msg = validate_text_input(user_input, "Question", min_len=3, max_len=500)
        if not is_valid:
            st.warning(error_msg)
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = route_and_answer(user_input)
                st.write(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    if st.session_state.chat_messages:
        chat_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_messages
        )
        st.download_button(
            "⬇️ Export chat (.txt)", chat_text, file_name="chat_history.txt", mime="text/plain"
        )

# ===================== TAB 2: WEATHER =====================
with tab2:
    st.subheader("Weather Lookup")
    city_input = st.text_input("City name", placeholder="e.g. Goa")

    if st.button("Get Weather", type="primary"):
        is_valid, error_msg = validate_text_input(city_input, "City name", min_len=2, max_len=100)
        if not is_valid:
            st.warning(error_msg)
        else:
            with st.spinner("Fetching weather..."):
                result, error = run_weather_lookup(city_input)
            if error:
                st.error(error)
                st.session_state.last_weather = None
            else:
                st.session_state.last_weather = result

    if st.session_state.last_weather:
        w = st.session_state.last_weather
        with st.container(border=True):
            st.subheader(f"{w['city']}, {w['country']}")
            st.caption(w["description"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{w['temp']} °C")
            c2.metric("Feels like", f"{w['feels_like']} °C")
            c3.metric("Humidity", f"{w['humidity']}%")
            st.caption(f"Wind speed: {w['wind']} m/s")

# ===================== TAB 3: FLIGHTS =====================
with tab3:
    st.subheader("Search Flights")
    st.caption("Aviationstack free API — enter 3-letter airport codes, e.g. DEL, BOM, GOI, JFK")
    col1, col2 = st.columns(2)
    with col1:
        dep_iata = st.text_input("Departure airport (IATA code)", placeholder="DEL").upper()
    with col2:
        arr_iata = st.text_input("Arrival airport (IATA code)", placeholder="BOM").upper()

    if st.button("Search Flights"):
        valid_dep, err1 = validate_iata_code(dep_iata, "Departure code")
        valid_arr, err2 = validate_iata_code(arr_iata, "Arrival code")
        if not valid_dep:
            st.warning(err1)
        elif not valid_arr:
            st.warning(err2)
        elif dep_iata == arr_iata:
            st.warning("Departure aur arrival airport same nahi ho sakte.")
        else:
            with st.spinner("Searching flights..."):
                flights, error = search_flights(dep_iata, arr_iata)
                save_search("flight", f"{dep_iata} -> {arr_iata}")
            if error:
                st.error(f"Flight search error: {error}")
                st.session_state.last_flights = None
            elif not flights:
                st.info("Is route ke liye koi flight nahi mili.")
                st.session_state.last_flights = None
            else:
                st.session_state.last_flights = flights

    if st.session_state.last_flights:
        for f in st.session_state.last_flights:
            with st.container(border=True):
                c1, c2 = st.columns([1, 5])
                c1.markdown("<div style='font-size:28px'>✈️</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{f['airline']} — {f['flight_number']}**")
                    st.write(f"{f['departure_airport']} ({f['departure_time']}) → "
                             f"{f['arrival_airport']} ({f['arrival_time']})")
                    st.caption(f"Status: {f['status']}")

        flights_df = pd.DataFrame(st.session_state.last_flights)
        st.download_button(
            "⬇️ Export results (.csv)",
            flights_df.to_csv(index=False),
            file_name=f"flights_{dep_iata}_{arr_iata}.csv",
            mime="text/csv",
        )

# ===================== TAB 4: ITINERARY =====================
with tab4:
    st.subheader("Generate a Travel Itinerary")
    destination = st.text_input("Destination", placeholder="e.g. Goa")
    days = st.number_input("Number of days", min_value=1, max_value=30, value=3)
    interests = st.text_input("Interests (optional)", placeholder="e.g. beaches, food, nightlife")

    if st.button("Generate Itinerary"):
        is_valid, error_msg = validate_text_input(destination, "Destination", min_len=2, max_len=100)
        if not is_valid:
            st.warning(error_msg)
        else:
            with st.spinner("Generating itinerary..."):
                itinerary = generate_itinerary(destination, int(days), interests)
                save_search("itinerary", f"{destination} - {days} days")
            st.session_state.last_itinerary = itinerary
            st.session_state.last_itinerary_dest = destination

    if st.session_state.last_itinerary:
        itinerary = st.session_state.last_itinerary
        blocks = itinerary.split("## ")
        rendered_any = False
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if heading.lower().startswith("day"):
                rendered_any = True
                with st.expander(f"📅 {heading}", expanded=True):
                    st.markdown(body if body else block)
        if not rendered_any:
            # fallback: model didn't use the "## Day N" format, just show it as-is
            st.markdown(itinerary)

        st.download_button(
            "⬇️ Download itinerary (.txt)",
            itinerary,
            file_name=f"{st.session_state.last_itinerary_dest or 'itinerary'}_itinerary.txt",
        )

# ===================== TAB 5: SEARCH HISTORY =====================
with tab5:
    st.subheader("Recent searches")
    history = get_searches()
    if not history:
        st.write("Koi search history nahi hai abhi tak.")
    else:
        history_df = pd.DataFrame(history, columns=["Type", "Details", "Timestamp"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.caption(
        "Note: Streamlit Cloud pe storage temporary hai — reboot/redeploy hone par history reset ho sakti hai."
    )
