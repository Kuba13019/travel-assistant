import os
import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️")
st.title("✈️ AI Travel Assistant")
st.caption("RAG chatbot + Web Search + Weather API — powered by LangChain")

# ---------- API keys (Streamlit Cloud secrets, or local env vars) ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.environ.get("OPENWEATHER_API_KEY", ""))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY set nahi hai. Streamlit Cloud app settings -> Secrets me add karo.")
    st.stop()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

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

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        st.sidebar.success(f"Document processed! ({len(chunks)} chunks)")
    except Exception as e:
        st.sidebar.error(f"Document process karne me error: {e}")

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()


def get_weather(city: str) -> str:
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


tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Current events, prices, ya live/recent travel info dhoondne ke liye use karo.",
    ),
    Tool(
        name="WeatherLookup",
        func=get_weather,
        description="Kisi city ka current weather jaanne ke liye use karo. Input sirf city ka naam ho.",
    ),
]

if st.session_state.vectorstore is not None:
    def doc_qa(query: str) -> str:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        return qa_chain.run(query)

    tools.append(
        Tool(
            name="TravelDocsQA",
            func=doc_qa,
            description="Uploaded travel guide document se related sawaal answer karne ke liye use karo.",
        )
    )

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

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
        with st.spinner("Soch raha hoon..."):
            try:
                response = agent.run(user_input)
            except Exception as e:
                response = f"Sorry, kuch galat ho gaya: {e}"
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
