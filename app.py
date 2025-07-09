# app.py

import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ✅ Load Gemini API key from environment variable
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("❌ Error: GOOGLE_API_KEY not set in environment.")
    exit()

# ✅ Gemini Model
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# ✅ Model & storage
model = SentenceTransformer("all-MiniLM-L6-v2")
all_chunks = []
metadata_list = []

# ✅ Extract PDF text
def extract_text_chunks(pdf_file_path, max_len=300):
    doc = fitz.open(pdf_file_path)
    chunks = []

    for page in doc:
        text = page.get_text()
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:
                chunks.append(para[:max_len])
    doc.close()
    return chunks

# ✅ Metadata extractors
def extract_year_from_filename(name):
    match = re.search(r'20\d{2}', name)
    return match.group(0) if match else "Unknown"

def detect_dept_from_text(text):
    departments = ['CSE', 'ECE', 'IT', 'MCA', 'MAE', 'AIDS', 'VLSI', 'BBA', 'MBA']
    for dept in departments:
        if dept in text.upper():
            return dept
    return "General"

def detect_batch_from_text(text):
    text = text.replace("–", "-").lower()
    patterns = [
        r'\b20\d{2}-20\d{2}\b',
        r'\bclass of (\d{4})\b',
        r'\bfor the year (\d{4})\b',
        r'\b(\d{4}) batch\b',
        r'\bgraduating batch of (\d{4})\b',
        r'\b(\d{4})\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0) if len(match.groups()) == 0 else match.group(1)
    return "Unknown"

def auto_extract_filters_from_query(query):
    year_match = re.search(r'20\d{2}', query)
    year = year_match.group(0) if year_match else "All"

    departments = ['CSE', 'ECE', 'IT', 'MCA', 'MAE', 'AIDS', 'VLSI', 'BBA', 'MBA']
    dept = "All"
    for d in departments:
        if d in query.upper():
            dept = d
            break

    return year, dept

# ✅ Load PDFs & build FAISS index
def build_index_from_pdfs(pdf_folder_path="./Placement Data"):
    global all_chunks, metadata_list, faiss_index

    if not os.path.exists(pdf_folder_path):
        print(f"❌ Error: Folder not found: {pdf_folder_path}")
        exit()

    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDFs found in {pdf_folder_path}")
        exit()

    for filename in pdf_files:
        file_path = os.path.join(pdf_folder_path, filename)
        print(f"📄 Loading: {filename}")
        year = extract_year_from_filename(filename)

        try:
            chunks = extract_text_chunks(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        for chunk in chunks:
            dept = detect_dept_from_text(chunk)
            batch = detect_batch_from_text(chunk)
            all_chunks.append(chunk)
            metadata_list.append({
                "text": chunk,
                "source": filename,
                "year": year,
                "dept": dept,
                "batch": batch
            })

    if not all_chunks:
        print("❌ No text extracted. Exiting.")
        exit()

    embeddings = model.encode(all_chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))

# ✅ Search relevant chunks
def get_top_chunks(query, year_filter=None, dept_filter=None, k=10):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata_list):
            meta = metadata_list[idx]
            if (year_filter in [None, "All", meta["year"]]) and (dept_filter in [None, "All", meta["dept"]]):
                results.append(meta)
        else:
            print(f"⚠️ Invalid index {idx} from FAISS.")
    return results

# ✅ Gemini answer generation
def get_gemini_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "⚠️ Could not find relevant information to answer your question."

    context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""
You are a placement assistant bot. Based on the placement data provided in the context, answer the user's question accurately and concisely.
If the context does not contain enough information, say you cannot answer based on the data.

Context:
{context}

User Question: {query}
Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "❌ An error occurred while generating the answer."

# ✅ Command line interaction
def main():
    print("\n🟢 You can now ask placement-related questions (type 'exit' to stop)\n")
    while True:
        try:
            query = input("💬 Your question: ").strip()
            if query.lower() in ['exit', 'quit', 'stop']:
                print("👋 Chat ended.")
                break

            if not query:
                print("Please enter a question.")
                continue

            year_filter, dept_filter = auto_extract_filters_from_query(query)
            print(f"🔎 Filters → Year: {year_filter} | Dept: {dept_filter}")

            top_chunks = get_top_chunks(query, year_filter=year_filter, dept_filter=dept_filter, k=20)

            if not top_chunks:
                print("⚠️ No relevant data found.")
                continue

            answer = get_gemini_answer(query, top_chunks)
            print("\n🔹 Answer:\n", answer)
            print("\n📄 Sources:")
            for i, chunk in enumerate(top_chunks[:min(len(top_chunks), 5)]):
                print(f"- Source {i+1}: {chunk.get('source', 'N/A')} | Year: {chunk.get('year', 'N/A')} | Dept: {chunk.get('dept', 'N/A')} | Batch: {chunk.get('batch', 'N/A')}")
            if len(top_chunks) > 5:
                print(f"... and {len(top_chunks) - 5} more result(s).")
            print("\n----------------------------------------\n")

        except EOFError:
            print("\n👋 Chat ended (Input closed).")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")

# ✅ Load index and optionally run CLI
if __name__ == "__main__":
    build_index_from_pdfs()
    main()
