import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from serpapi import GoogleSearch

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

SERPAPI_KEY = "<YOUR_SERPAPI_KEY>"

def fetch_citation_count_google_scholar(title):
    params = {
        "engine": "google_scholar",
        "q": title,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "organic_results" in results and len(results["organic_results"]) > 0:
        item = results["organic_results"][0]
        citations = item.get("inline_links", {}).get("cited_by", {}).get("total", 0)
        return int(citations) if citations else 0
    return 0

def search_arxiv(query, max_results=10):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    print(f"\nSearching arXiv for: '{query}'...\n")
    response = requests.get(ARXIV_API_URL, params=params)
    if response.status_code != 200:
        print("Failed to fetch results from arXiv.")
        return []

    root = ET.fromstring(response.content)
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
    results = []

    for entry in root.findall('arxiv:entry', ns):
        title = entry.find('arxiv:title', ns).text.strip().replace('\n', ' ')
        authors = [author.find('arxiv:name', ns).text for author in entry.findall('arxiv:author', ns)]
        summary = entry.find('arxiv:summary', ns).text.strip().replace('\n', ' ')
        published = entry.find('arxiv:published', ns).text

        citation_count = fetch_citation_count_google_scholar(title)

        results.append({
            "title": title,
            "authors": authors,
            "summary": summary,
            "published": published,
            "citations": citation_count
        })

    def rank_score(paper):
        from datetime import datetime
        year_weight = 0
        try:
            year = int(paper['published'][:4])
            year_weight = (datetime.now().year - year)
        except:
            pass
        return paper['citations'] - (0.5 * year_weight)

    results.sort(key=rank_score, reverse=True)
    return results

def build_faiss_index(papers):
    chunks = [p['summary'] for p in papers]
    embeddings = embed_model.encode(chunks)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    SESSION["faiss_index"] = index
    SESSION["embeddings"] = embeddings
    SESSION["chunks"] = chunks

def handle_initial_query(query):
    print(f"\n[+] Searching papers for: '{query}'")
    papers = search_arxiv(query, max_results=10)
    build_faiss_index(papers)
    SESSION["query"] = query
    SESSION["papers"] = papers
    print("[+] Top papers loaded into memory.")
    for i, p in enumerate(papers):
        print(f"  {i+1}. {p['title']} (Citations: {p['citations']})")

def answer_followup(query):
    print(f"\n[+] Answering follow-up: '{query}'")
    q_emb = embed_model.encode([query])
    D, I = SESSION["faiss_index"].search(np.array(q_emb), k=3)
    relevant_chunks = [SESSION["chunks"][i] for i in I[0]]
    papers_used = [SESSION["papers"][i] for i in I[0]]

    context = "\n\n".join(relevant_chunks)
    prompt = f"You are a research assistant.\n\nContext:\n{context}\n\nQ: {query}\nA:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5
    )
    print("\n[AI]:", response.choices[0].text.strip())

    print("\nSources:")
    for paper in papers_used:
        print(f"- {paper['title']} ({paper['published']}, Citations: {paper['citations']})")

if __name__ == "__main__":
    openai.api_key = "<YOUR_OPENAI_API_KEY>"
    
    while True:
        if SESSION["query"] is None:
            user_query = input("\nEnter your research topic: ")
            handle_initial_query(user_query)
        else:
            followup = input("\nAsk a follow-up question (or type 'new' to start over): ")
            if followup.lower() == 'new':
                SESSION["query"] = None
                continue
            answer_followup(followup)