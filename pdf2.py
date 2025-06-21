# Advanced RAG Research Assistant using Google Search + arXiv + PDF extraction

import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
from PyPDF2 import PdfReader
from get_llm_response import get_response
# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SESSION = {
    "query": None,
    "papers": [],
    "faiss_index": None,
    "embeddings": [],
    "chunks": []
}

SERPAPI_KEY = ""

def search_google_scholar(query, max_results=2):
    print(f"\n[+] Searching Google for top papers related to: '{query}'")
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_KEY
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        print("[!] Failed to fetch results from SerpAPI.")
        return []

    results = response.json()
    papers = []

    if "organic_results" in results:
        for item in results["organic_results"][:max_results]:
            title = item.get("title")
            link = item.get("link")
            citations = item.get("inline_links", {}).get("cited_by", {}).get("total", 0)
            papers.append({
                "title": title,
                "link": link,
                "citations": citations
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
    if entry is not None:
        for link in entry.findall('arxiv:link', ns):
            if link.attrib.get('type') == 'application/pdf':
                return link.attrib['href']
    return None

def download_and_extract_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        filename = "temp_paper.pdf"
        with open(filename, "wb") as f:
            f.write(response.content)

        reader = PdfReader(filename)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        os.remove(filename)
        return text
    except Exception as e:
        print(f"[!] Failed to extract PDF text: {e}")
        return ""

def build_faiss_index_from_google(query):
    results = search_google_scholar(query)
    papers = []
    texts = []
    for paper in results:
        print(f"  [+] Processing: {paper['title']}")
        pdf_url = search_arxiv_for_pdf(paper['title'])
        if pdf_url:
            paper_text = download_and_extract_text(pdf_url)
            if paper_text.strip():
                papers.append(paper)
                texts.append(paper_text)
            else:
                print("     [!] Empty text extracted.")
        else:
            print("     [!] No arXiv PDF found.")

    if not texts:
        print("[!] No valid papers were processed. FAISS index not built.")
        return

    embeddings = embed_model.encode(texts)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    SESSION["query"] = query
    SESSION["papers"] = papers
    SESSION["chunks"] = texts
    SESSION["embeddings"] = embeddings
    SESSION["faiss_index"] = index
    print(f"[+] Loaded {len(papers)} papers into FAISS database.")

def answer_followup(query):
    print(f"\n[+] Answering: '{query}'")
    q_emb = embed_model.encode([query])
    D, I = SESSION["faiss_index"].search(np.array(q_emb), k=3)
    relevant_chunks = [SESSION["chunks"][i] for i in I[0]]
    papers_used = [SESSION["papers"][i] for i in I[0]]

    context = "\n\n".join(relevant_chunks[:1][:1000])
    prompt = f"You are a research assistant.\n\nContext:\n{context}\n\n Q::"
    print("AAAAAAAAAAAAAAAAAA", len(prompt))
    #print(context)
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=300,
    #     temperature=0.5
    # )
    # print("\n[AI]:", response.choices[0].text.strip())
    response = get_response(prompt, query)
    print("RESPONSE__________________\n", response)
    print("\nSources:")
    for paper in papers_used:
        print(f"- {paper['title']} (Citations: {paper['citations']}, Link: {paper['link']})")

if __name__ == "__main__":
    while True:
        if SESSION["query"] is None:
            user_query = input("\nEnter your research topic: ")
            build_faiss_index_from_google(user_query)
        else:
            followup = input("\nAsk a follow-up question (or type 'new' to start over): ")
            if followup.lower() == 'new':
                SESSION["query"] = None
                continue
            answer_followup(followup)
