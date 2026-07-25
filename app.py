import os
import json
import re
import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️")
st.title("✈️ AI Travel Assistant")
st.caption("RAG chatbot + Web Search + Weather API — powered by LangChain + Groq")

# ---------- API keys (Streamlit Cloud secrets, or local env vars) ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.environ.get("OPENWEATHER_API_KEY", ""))

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY set nahi hai. Streamlit Cloud app settings -> Secrets me add karo.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)


@st.cache_resource
def load_embeddings():
    return FakeEmbeddings(size=768)


embeddings_model = load_embeddings()

# ---------- Sidebar: document upload (RAG) ----------
st.sidebar.header("📄 Travel Document Upload")
uploaded_file = st.sidebar.file_uploader("PDF ya TXT travel guide upload karo", type=["pdf", "txt"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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

# ---------- Tool functions ----------
search_tool_runner = DuckDuckGoSearchRun()


def run_web_search(query: str) -> str:
    try:
        return search_tool_runner.run(query)
    except Exception as e:
        return f"Web search error: {e}"


def run_weather_lookup(city: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "Weather API key configured nahi hai."
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("cod") != 200:
            return f"Sorry, '{city}' ka weather nahi mil paya. ({data.get('message', 'unknown error')})"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"{city} mein abhi {temp}°C hai, aasman: {desc}"
    except requests.exceptions.Timeout:
        return "Weather service timeout ho gaya, thodi der baad try karo."
    except Exception as e:
        return f"Weather fetch error: {e}"


def run_doc_qa(query: str) -> str:
    if st.session_state.vectorstore is None:
        return "Koi document upload nahi kiya gaya hai."
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    return qa_chain.run(query)


# ---------- Manual router (avoids Groq's native function-calling parsing bug) ----------
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
    tool_name = "none"
    tool_input = user_input
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
        tool_result = run_weather_lookup(tool_input)
    elif tool_name == "TravelDocsQA" and doc_available:
        tool_result = run_doc_qa(tool_input)
    else:
        tool_result = None

    if tool_result is None:
        final_prompt = f"Answer this travel-related question helpfully and concisely: {user_input}"
    else:
        final_prompt = (
            f"User question: {user_input}\n\n"
            f"Tool used: {tool_name}\n"
            f"Tool result: {tool_result}\n\n"
            f"Using the tool result above, give a natural, helpful, concise answer to the user's question."
        )

    try:
        final_response = llm.invoke(final_prompt).content
        return final_response
    except Exception as e:
        return f"Sorry, kuch galat ho gaya: {e}"


# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Apni trip ke baare me kuch bhi pucho...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = route_and_answer(user_input)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
