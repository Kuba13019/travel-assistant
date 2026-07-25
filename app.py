import os
import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
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

# ---------- Tools (using @tool decorator for correct schema with tool-calling agent) ----------
search_tool_runner = DuckDuckGoSearchRun()


@tool
def WebSearch(query: str) -> str:
    """Search the web for current events, prices, or any live/recent travel information."""
    return search_tool_runner.run(query)


@tool
def WeatherLookup(city: str) -> str:
    """Get the current weather for a city. Input should be just the city name, e.g. 'Goa'."""
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


tools = [WebSearch, WeatherLookup]

if st.session_state.vectorstore is not None:
    @tool
    def TravelDocsQA(query: str) -> str:
        """Answer questions using the uploaded travel guide document."""
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        return qa_chain.run(query)

    tools.append(TravelDocsQA)

# ---------- Tool-calling agent (reliable with Groq models) ----------
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Use the available tools to answer the "
               "user's travel-related questions accurately and concisely."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
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
       with st.spinner("Thinking..."):
    response = None
    last_error = None
    for attempt in range(2):
        try:
            result = agent_executor.invoke({"input": user_input})
            response = result["output"]
            break
        except Exception as e:
            last_error = e
    if response is None:
        response = f"Sorry, failed after retry: {last_error}"
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
