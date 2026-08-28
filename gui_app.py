import os
import json
import uuid
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 1. Environment & Initialization
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Holybot 2.0 | World Beyond Money",
    page_icon="🤖",
    layout="wide"
)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY. Add it to your `.env` file or Streamlit Secrets.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Vector DB & Retriever Caching
# -----------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "holybot_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"❌ Failed to load ChromaDB. Did you run `ingest.py` first? Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Slider Mapping
# -----------------------------------------------------------------------------
st.sidebar.title("🤖 Holybot 2.0 Controls")

SLIDER_MAP = {
    0: {
        "label": "Concise", 
        "k": 2, 
        "instruction": "Keep your response sharp, direct, and under 120 words."
    },
    1: {
        "label": "Balanced", 
        "k": 3, 
        "instruction": "Provide a balanced response (~200 words) combining pragmatic systems analysis with key WBM principles."
    },
    2: {
        "label": "Detailed", 
        "k": 4, 
        "instruction": "Provide a detailed breakdown (~300 words) with analytical context and WBM theoretical depth."
    }
}

slider_val = st.sidebar.select_slider(
    "Speed vs. Depth Mode",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: SLIDER_MAP[x]["label"],
    help="Adjusts chunk retrieval depth (k) and persona target response length."
)

current_mode = SLIDER_MAP[slider_val]
enable_logging = st.sidebar.checkbox("Enable Anonymous Session Logging", value=True)

# -----------------------------------------------------------------------------
# 4. Account Model Resolver & LLM Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_active_groq_model(api_key: str) -> str:
    env_override = os.getenv("GROQ_MODEL")
    if env_override:
        return env_override

    try:
        client = Groq(api_key=api_key)
        models_data = client.models.list()
        active_ids = [m.id for m in models_data.data if getattr(m, 'active', True)]
        
        # Priority list matching your exact account deployment IDs
        preferences = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound"
        ]
        
        for pref in preferences:
            if pref in active_ids:
                return pref
                
        return active_ids[0] if active_ids else "openai/gpt-oss-120b"
        
    except Exception:
        return "openai/gpt-oss-120b"

selected_model = get_active_groq_model(groq_api_key)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.caption(f"**Active Model:** `{selected_model}`")
st.sidebar.caption(f"**Retriever Depth (k):** `{current_mode['k']} chunks`")

SYSTEM_PROMPT_TEMPLATE = """You are Holybot 2.0 (HB2), an interactive AI assistant for the World Beyond Money (WBM) initiative, speaking on behalf of Holyman's written work.

Persona & Tone:
- Speak with dry English wit, pragmatic systems perspective, and iconoclastic edge.
- Zero fluff or buzzwords. Use concise bullet points where applicable.

Response Constraints:
{length_instruction}

Context from Holyman's Corpus:
{context}

Recent Chat History:
{chat_history}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=selected_model,
    temperature=0.3,
    max_tokens=1024
)

retriever = vectorstore.as_retriever(search_kwargs={"k": current_mode["k"]})

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)

def get_recent_history():
    if "messages" not in st.session_state or not st.session_state.messages:
        return "None"
    recent = st.session_state.messages[-4:]
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])

# RAG Chain Assembly
rag_chain = (
    {
        "context": retriever | format_docs,
        "chat_history": lambda _: get_recent_history(),
        "question": RunnablePassthrough(),
        "length_instruction": lambda _: current_mode["instruction"]
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# -----------------------------------------------------------------------------
# 5. Logging Helper
# -----------------------------------------------------------------------------
def log_interaction(prompt: str, response: str, mode_label: str):
    if not enable_logging:
        return
    
    os.makedirs("./logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": st.session_state.session_id,
        "mode": mode_label,
        "user_prompt": prompt,
        "bot_response": response
    }
    
    try:
        with open("./logs/chat_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        st.sidebar.warning(f"Logging failed: {e}")

# -----------------------------------------------------------------------------
# 6. Streamlit Chat Interface
# -----------------------------------------------------------------------------
st.title("Holybot 2.0")
st.caption("Interactive AI Assistant for the World Beyond Money Initiative")

# Render persistent message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_input := st.chat_input("Ask about WBM, systemic transition, or Holyman's works..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing systems context..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                log_interaction(user_input, response, current_mode["label"])
                
            except Exception as e:
                st.error(f"Error querying Groq LLM endpoint: {e}")