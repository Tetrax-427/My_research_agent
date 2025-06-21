import streamlit as st
import requests
import xml.etree.ElementTree as ET
import numpy as np
import os
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from get_llm_response import get_response  # Custom LLM

# Config
from dotenv import load_dotenv
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
ARXIV_API_URL = "http://export.arxiv.org/api/query"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# State
if "query" not in st.session_state:
    st.session_state.query = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "papers" not in st.session_state:
    st.session_state.papers = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []
if "model_name" not in st.session_state:
    st.session_state.model_name = "llama-3.3-70b-versatile"
if "temp" not in st.session_state:
    st.session_state.temp = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 300

# Functions
def search_google_scholar(query, max_results=20):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_KEY
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        return []
    results = response.json()
    papers = []
    for item in results.get("organic_results", [])[:max_results]:
        papers.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "citations": item.get("inline_links", {}).get("cited_by", {}).get("total", 0)
        })
    return papers

def search_arxiv_for_pdf(title):
    params = {
        "search_query": f"ti:\"{title}\"",
        "start": 0,
        "max_results": 1
    }
    response = requests.get(ARXIV_API_URL, params=params)
    if response.status_code != 200:
        return None
    root = ET.fromstring(response.content)
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
    entry = root.find('arxiv:entry', ns)
    if entry:
        for link in entry.findall('arxiv:link', ns):
            if link.attrib.get('type') == 'application/pdf':
                return link.attrib['href']
    return None

def download_and_extract_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        filename = "temp.pdf"
        with open(filename, "wb") as f:
            f.write(response.content)
        reader = PdfReader(filename)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        os.remove(filename)
        return text
    except:
        return ""

def build_faiss_index(query):
    results = search_google_scholar(query)
    papers, texts = [], []
    for paper in results:
        pdf_url = search_arxiv_for_pdf(paper['title'])
        if pdf_url:
            paper_text = download_and_extract_text(pdf_url)
            if paper_text.strip():
                papers.append(paper)
                texts.append(paper_text)
    if not texts:
        return False
    embeddings = embed_model.encode(texts)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    st.session_state.faiss_index = index
    st.session_state.query = query
    st.session_state.papers = papers
    st.session_state.chunks = texts
    st.session_state.history = []
    return True

def answer_query(user_query):
    q_emb = embed_model.encode([user_query])
    D, I = st.session_state.faiss_index.search(np.array(q_emb), k=3)
    context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
    context =context[:10000]
    system_prompt = "You are a research assistant. Use the context from papers to answer the query clearly."
    return get_response(
        system_prompt,
        user_query + "\n\nContext:\n" + context,
        model=st.session_state.model_name,
        temperature=st.session_state.temp,
        top_p=st.session_state.top_p,
        max_new_tokens=st.session_state.max_tokens
    )

# UI Layout
st.set_page_config(page_title="RAG Research Assistant", layout="wide")

with st.sidebar:
    st.title("🔧 Tools")
    st.session_state.model_name = st.selectbox("Select LLM Model", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","qwen-qwq-32b","qwen/qwen3-32b","deepseek-r1-distill-llama-70b","mistral-saba-24b"])

    with st.expander("LLM Options"):
        st.session_state.temp = st.slider("Temperature", 0.0, 1.0, st.session_state.temp, step=0.05)
        st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, step=0.05)
        st.session_state.max_tokens = st.number_input("Max New Tokens", min_value=50, max_value=2048, value=st.session_state.max_tokens)

st.title("📚 Research Assistant")
topic = st.text_input("Enter Research Topic to Start New Session", value="" if not st.session_state.query else st.session_state.query)

if topic and topic != st.session_state.query:
    with st.spinner("Loading and indexing papers..."):
        success = build_faiss_index(topic)
        if not success:
            st.error("No valid papers found. Try a different topic.")
        else:
            st.success("Knowledge base built! You can now ask questions.")

if st.session_state.faiss_index:
    if st.session_state.history:
        #st.markdown("### Chat History")
        for q, a in st.session_state.history:
            st.chat_message("user").markdown(q)
            st.chat_message("assistant").markdown(a)

    #st.subheader("💬 Chat with Your Research Assistant")
    user_input = st.chat_input("Ask a question about your research topic")
    if user_input:
        with st.spinner("Thinking..."):
            response = answer_query(user_input)
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(response)
        st.session_state.history.append((user_input, response))
