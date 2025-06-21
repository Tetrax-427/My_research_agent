# cli_index_builder.py
import os
import json
import faiss
import requests
import xml.etree.ElementTree as ET
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


ARXIV_API_URL = "http://export.arxiv.org/api/query"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = "vector_db"

os.makedirs(DB_PATH, exist_ok=True)

def search_google_scholar(query, max_results=100):
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

def build_index_for_topic(topic):
    print(f"Searching for papers on '{topic}'...")
    results = search_google_scholar(topic)
    papers, texts = [], []
    for paper in results:
        pdf_url = search_arxiv_for_pdf(paper['title'])
        if pdf_url:
            paper_text = download_and_extract_text(pdf_url)
            if paper_text.strip():
                papers.append(paper)
                texts.append(paper_text)
    if not texts:
        print("No valid papers found.")
        return False
    embeddings = EMBED_MODEL.encode(texts)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, f"{DB_PATH}/{topic}.index")
    with open(f"{DB_PATH}/{topic}.json", "w") as f:
        json.dump({"papers": papers, "chunks": texts}, f)
    print(f"Index built and saved for topic: '{topic}'")
    return True

if __name__ == "__main__":
    topic = input("Enter topic: ").strip()
    build_index_for_topic(topic)
