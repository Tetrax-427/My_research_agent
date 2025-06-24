# app.py (Streamlit UI)
import os
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from get_llm_response import get_response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16).eval()

DB_PATH = "vector_db"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
token_false_id = qwen_tokenizer.convert_tokens_to_ids("no")
token_true_id = qwen_tokenizer.convert_tokens_to_ids("yes")
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = qwen_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = qwen_tokenizer.encode(suffix, add_special_tokens=False)
max_length = 1024

print("Models LOADED...")

def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs):
    inputs = qwen_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = qwen_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(qwen_model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    batch_scores = qwen_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def rerank_chunks(query, chunks, top_k=3):
    task = "Given a web search query, retrieve relevant passages that answer the query"
    pairs = [format_instruction(task, query, chunk) for chunk in chunks]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

st.set_page_config(page_title="RAG Research Assistant", layout="wide")

# UI State init
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
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

# Sidebar controls
with st.sidebar:
    st.title("🔧 Tools")
    st.session_state.model_name = st.selectbox("Select LLM Model", 
                                               ["llama-3.3-70b-versatile",
                                                "llama-3.1-8b-instant",
                                                "qwen-qwq-32b",
                                                "qwen/qwen3-32b",
                                                "deepseek-r1-distill-llama-70b",
                                                "mistral-saba-24b"])
    with st.expander("LLM Options"):
        st.session_state.temp = st.slider("Temperature", 0.0, 1.0, st.session_state.temp, step=0.05)
        st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, step=0.05)
        st.session_state.max_tokens = st.number_input("Max New Tokens", min_value=50, max_value=2048, value=st.session_state.max_tokens)

# Load available topics
available_topics = [f.split(".")[0] for f in os.listdir(DB_PATH) if f.endswith(".index")]
st.title("📚 Research Assistant")
topic = st.selectbox("Select a preloaded topic:", [""] + available_topics)

if topic:
    index_path = os.path.join(DB_PATH, topic + ".index")
    json_path = os.path.join(DB_PATH, topic + ".json")
    if os.path.exists(index_path) and os.path.exists(json_path):
        st.session_state.faiss_index = faiss.read_index(index_path)
        with open(json_path) as f:
            data = json.load(f)
            st.session_state.chunks = data["chunks"]
        st.session_state.history = []
        st.success(f"Topic '{topic}' loaded successfully!")
    else:
        st.error("Topic files missing.")
else:
    st.error("Topic files missing.")


# Query handler
def answer_query(user_query):
    q_emb = EMBED_MODEL.encode([user_query])
    D, I = st.session_state.faiss_index.search(np.array(q_emb), k=3)
    retrieved_chunks = [st.session_state.chunks[i][:1000] for i in I[0]]
    reranked_chunks = rerank_chunks(user_query, retrieved_chunks, top_k=10)
    context = "\n\n".join(reranked_chunks)
    context = context[:10000]
    system_prompt = "You are a research assistant. Use the context from papers to answer the query clearly."
    return get_response(
        system_prompt,
        user_query + "\n\nContext:\n" + context,
        model=st.session_state.model_name,
        temperature=st.session_state.temp,
        top_p=st.session_state.top_p,
        max_new_tokens=st.session_state.max_tokens
    )

if st.session_state.faiss_index:
    if st.session_state.history:
        st.markdown("### Chat History")
        for q, a in st.session_state.history:
            st.chat_message("user").markdown(q)
            st.chat_message("assistant").markdown(a)

    st.subheader("💬 Chat with Your Research Assistant")
    user_input = st.chat_input("Ask a question about your research topic")
    if user_input:
        with st.spinner("Thinking..."):
            response = answer_query(user_input)
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(response)
        st.session_state.history.append((user_input, response))
