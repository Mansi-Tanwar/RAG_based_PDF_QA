# app.py
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import tempfile
import os

# Setup Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

# Load embedder once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Utility: Extract text from uploaded PDF
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Utility: Split into chunks
def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Main RAG QA Function
def rag_qa(query, chunks, index):
    try:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), 3)
        context = "\n".join([chunks[i] for i in indices[0]])
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {e}"

# Streamlit UI
st.set_page_config(page_title="PDF Q&A System", layout="centered")
st.title("📄 PDF Question Answering using RAG")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask your question")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("🔍 Extracting and embedding..."):
        text = extract_text_from_pdf(tmp_path)
        if text:
            chunks = split_text_into_chunks(text)
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            st.success("✅ PDF processed. Ask away!")

    if question:
        with st.spinner("💭 Thinking..."):
            answer = rag_qa(question, chunks, index)
            st.markdown(f"**🧠 Answer:** {answer}")
