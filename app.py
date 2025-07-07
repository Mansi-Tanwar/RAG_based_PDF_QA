import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import tempfile
import traceback

# 🎯 Set page config
st.set_page_config(page_title="📄 RAG PDF Q&A", layout="centered")

# 🎉 Title and UI
st.title("📄 PDF Question Answering using RAG + Gemini")
st.markdown("Upload a PDF and ask questions. The system uses Gemini 2.5 Flash with semantic search.")

# 📌 Setup Gemini API (set this in Streamlit Secrets)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
except Exception as e:
    st.error("❌ Gemini API key missing or invalid. Please check your Streamlit secrets.")
    st.stop()

# 🧠 Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🧾 File uploader and question input
uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"])
question = st.text_input("❓ Ask your question")

# 🔧 Utility functions
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def rag_qa(query, chunks, index):
    try:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), 3)
        context = "\n".join([chunks[i] for i in indices[0]])
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error("❌ Error generating answer")
        st.text(traceback.format_exc())
        return None

# 🧠 Embedding and Search
chunks, index = None, None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("📖 Reading and embedding the PDF..."):
        text = extract_text_from_pdf(tmp_path)
        if text:
            chunks = split_text_into_chunks(text)
            try:
                embeddings = embedder.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))
                st.success("✅ PDF processed successfully!")
            except Exception as e:
                st.error("❌ Error during embedding/indexing.")
                st.text(traceback.format_exc())

# 🤖 Q&A
if uploaded_file and question and chunks and index:
    with st.spinner("💭 Thinking..."):
        answer = rag_qa(question, chunks, index)
        if answer:
            st.markdown(f"**🧠 Answer:** {answer}")
elif question and not uploaded_file:
    st.warning("⚠️ Please upload a PDF first before asking questions.")
