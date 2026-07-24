# AI Travel Assistant

Ek AI-powered travel assistant jo document upload karke RAG-based sawal-jawab karta hai, aur Web Search + Weather API tools use karke live travel information deta hai. LangChain (agent architecture) + Streamlit (UI) + Streamlit Cloud (deployment) se banaya gaya hai.

## Features
- Document upload (PDF/TXT) + Retrieval-Augmented Generation (RAG)
- Web Search tool (DuckDuckGo)
- Weather API tool (OpenWeatherMap)
- Tool-using agent (LangChain ZERO_SHOT_REACT agent) - khud decide karta hai kaunsa tool use karna hai
- Streamlit chat interface

## Local setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

Local run ke liye `.streamlit/secrets.toml` file banao:
```toml
OPENAI_API_KEY = "your-key-here"
OPENWEATHER_API_KEY = "your-key-here"
```

## Deployment
Streamlit Cloud (https://streamlit.io/cloud) pe is repo ko connect karo, `app.py` ko main file set karo, aur app settings ke "Secrets" section mein upar wali dono keys add karo.

## Tech stack
- LangChain
- OpenAI GPT-3.5-turbo
- FAISS (vector store)
- Streamlit
