import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from serpapi import GoogleSearch
import os
from PyPDF2 import PdfReader
from get_llm_response import get_response
# Set your OpenAI API key here or export it as OPENAI_API_KEY environment variable

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


SESSION = {
    "query": None,
    "papers": [],
    "faiss_index": None,
    "embeddings": [],
    "chunks": []
}

def fetch_citation_count(title):
    try:
        params = {
            "query": title,
            "fields": "title,citationCount",
            "limit": 1
        }
        headers = {"User-Agent": "AI Research Assistant"}
        response = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0].get("citationCount", 0)
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
    return 0

def search_google_scholar(query, api_key, max_results=10):
    print(f"\nSearching Google Scholar for: '{query}' using SerpAPI...\n")
    
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    papers = []

    if "organic_results" not in results:
        print("No results found.")
        return []

    for item in results["organic_results"][:max_results]:
        paper = {
            "title": item.get("title", "No title"),
            "summary": item.get("snippet", "No summary available"),
            "link": item.get("link", ""),
            "citations": int(item.get("inline_links", {}).get("cited_by", {}).get("total", 0)),
            "published": "Unknown",  # Google Scholar doesn't always provide it
            "authors": item.get("publication_info", {}).get("authors", [])
        }
        papers.append(paper)

    return papers
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

        # Get citation count from Semantic Scholar
        citation_count = fetch_citation_count(title)

        results.append({
            "title": title,
            "authors": authors,
            "summary": summary,
            "published": published,
            "citations": citation_count
        })

    # Rank papers by citation count, newest papers get a slight boost
    def rank_score(paper):
        from datetime import datetime
        year = int(paper['published'][:4])
        current_year = datetime.now().year
        citation_score = paper['citations']
        recency_bonus = max(0, 5 - (current_year - year))  # Bonus for papers <= 5 years old
        return citation_score + recency_bonus

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

def summarize_text(text):
    if not openai.api_key:
        return "OpenAI API key not set. Cannot generate summary."

    prompt = (
        "Summarize the following research paper abstract in 2-3 sentences:\n\n"
        + text
        + "\n\nSummary:"
    )

    # try:
    #     response = openai.Completion.create(
    #         engine="text-davinci-003",
    #         prompt=prompt,
    #         max_tokens=150,
    #         temperature=0.5,
    #         top_p=1.0,
    #         n=1,
    #         stop=None,
    #     )
    #     summary = response.choices[0].text.strip()
    #     return summary
    # except Exception as e:
    #     return f"Error generating summary: {e}"
    return "Dummmyyyyyyy Text"

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

    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=300,
    #     temperature=0.5
    # )
    # print("\n[AI]:", response.choices[0].text.strip())

    print("\nSources:")
    for paper in papers_used:
        print(f"- {paper['title']} ({paper['published']}, Citations: {paper['citations']})")

def main():
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

main()
    
    