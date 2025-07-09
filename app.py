# app.py

import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import nltk
import collections
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Configure Gemini API key securely
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("❌ Error: GOOGLE_API_KEY not set in environment.")
    exit()

gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# ✅ Ensure nltk punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ✅ Global variables
model = SentenceTransformer("all-MiniLM-L6-v2")
all_chunks = []
metadata_list = []
faiss_index = None

# ✅ Embed using Gemini
def embed_text(texts):
    if not isinstance(texts, list):
        texts = [texts]
    if not texts or all(not t.strip() for t in texts):
        return np.array([])
    responses = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document",
        title="PlacementQA"
    )
    embeddings = responses["embedding"]
    if isinstance(embeddings[0], float):
        embeddings = [embeddings]
    return np.array(embeddings)

# ✅ Extract chunks from a single PDF
def extract_chunks_from_pdf(pdf_path, year, max_tokens=200):
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            page_text = []
            for b in blocks:
                txt = b[4].strip().replace('\n', ' ')
                if len(txt) < 20 or "thank you" in txt.lower():
                    continue
                page_text.append(txt)
            text = " ".join(page_text)
            sentences = sent_tokenize(text)
            current, count = [], 0
            for sent in sentences:
                tokens = sent.split()
                if count + len(tokens) > max_tokens and current:
                    chunk = " ".join(current)
                    chunks.append(f"YEAR: {year}\n{chunk}")
                    current, count = [], 0
                current.extend(tokens)
                count += len(tokens)
            if current:
                chunk = " ".join(current)
                chunks.append(f"YEAR: {year}\n{chunk}")
        doc.close()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    return chunks

# ✅ Get raw text from PDFs for year-based metric extraction
def build_raw_text_map(pdf_folder):
    raw_map = {}
    for fname in os.listdir(pdf_folder):
        if not fname.endswith(".pdf"):
            continue
        match = re.search(r'(\d{4})', fname)
        year = match.group(1) if match else None
        if not year:
            continue
        pdf_path = os.path.join(pdf_folder, fname)
        try:
            doc = fitz.open(pdf_path)
            page_texts = []
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))
                for b in blocks:
                    txt = b[4].strip().replace('\n', ' ')
                    if len(txt) < 20 or "thank you" in txt.lower():
                        continue
                    page_texts.append(txt)
            doc.close()
            raw_map[year] = " ".join(page_texts)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    return raw_map

# ✅ Parse placement metrics
def parse_year_metrics(raw_text_by_year):
    metrics = collections.defaultdict(dict)
    for year, text in raw_text_by_year.items():
        cleaned_text = text.replace('\n', ' ')
        avg = re.search(r'Average(?: Salary| CTC)?.*?(\d+\.?\d*)', cleaned_text, re.IGNORECASE)
        high = re.search(r'Highest(?: Salary| CTC)?.*?(\d+\.?\d*)', cleaned_text, re.IGNORECASE)
        recs = re.search(r'(?:companies|recruiters).{0,20}(\d{2,4})', cleaned_text, re.IGNORECASE)

        metrics[year]['average_salary'] = float(avg.group(1)) if avg else None
        metrics[year]['highest_salary'] = float(high.group(1)) if high else None
        metrics[year]['companies_visited'] = int(recs.group(1)) if recs else None
    return metrics

# ✅ Build FAISS index
def create_faiss_index(chunks):
    embeddings = embed_text(chunks)
    if embeddings.size == 0:
        return None, [], np.array([])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks, embeddings

# ✅ Retrieve top relevant chunks
def retrieve_chunks(query, index, chunks, embeddings, top_k=20):
    query_emb = embed_text(query)
    if query_emb.size == 0 or index is None:
        return ""
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    valid = [i for i in I[0] if i != -1 and i < len(chunks)]
    return "\n".join([chunks[i] for i in valid])

# ✅ Ask Gemini
def ask_gemini(question, context):
    if not context.strip():
        return "⚠️ I couldn't find enough context to answer that question."
    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
Context:
{context}

Question: {question}
Answer:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "❌ Gemini API error."

# ✅ Hybrid QA logic
def answer_question(query, index, chunks, embeddings, year_metrics):
    query_lower = query.lower()
    m_year = re.search(r"\b(20\d{2})\b", query_lower)
    year = m_year.group(1) if m_year else None

    if year and year in year_metrics:
        m = year_metrics[year]
        if "average" in query_lower and "salary" in query_lower and m.get("average_salary"):
            return f"The average salary in {year} was {m['average_salary']} LPA."
        if "highest" in query_lower and "salary" in query_lower and m.get("highest_salary"):
            return f"The highest salary in {year} was {m['highest_salary']} LPA."
        if any(k in query_lower for k in ["companies", "recruiters"]) and m.get("companies_visited"):
            return f"{m['companies_visited']} companies visited IGDTUW in {year}."

    context = retrieve_chunks(query, index, chunks, embeddings)
    return ask_gemini(query, context)

# ✅ Entry point
def build_and_run(pdf_folder_path):
    print(f"📁 Reading from: {pdf_folder_path}")
    raw_text_map = build_raw_text_map(pdf_folder_path)
    metrics = parse_year_metrics(raw_text_map)

    all_chunks = []
    for fname in os.listdir(pdf_folder_path):
        if not fname.lower().endswith(".pdf"):
            continue
        match = re.search(r'(\d{4})', fname)
        year = match.group(1) if match else None
        if year:
            print(f"→ Processing {fname}")
            all_chunks.extend(extract_chunks_from_pdf(os.path.join(pdf_folder_path, fname), year))

    index, chunk_texts, embs = create_faiss_index(all_chunks)
    if not chunk_texts:
        print("❌ No chunks created.")
        return

    while True:
        q = input("❓ Ask a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print("🧠 Answer:", answer_question(q, index, chunk_texts, embs, metrics))
        print()

# ✅ Local script execution (for CLI testing)
if __name__ == "__main__":
    build_and_run("./Placement Data")
