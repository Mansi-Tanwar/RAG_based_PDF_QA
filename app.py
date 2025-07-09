import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import tempfile
import traceback

# 🎯 Page config
st.set_page_config(page_title="📄 RAG PDF Q&A Chat", layout="centered")

# 🎉 Title
st.title("📄 PDF Chat using RAG + Gemini Flash 2.5")
st.markdown("Upload a PDF and chat with it using Gemini and RAG!")

# 📌 Gemini API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
except Exception as e:
    st.error("❌ Gemini API key missing or invalid. Please check your Streamlit secrets.")
    st.stop()

# 🧠 Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 📎 PDF upload
uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"])

# 🧠 Session states for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

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
        prompt = f"Use the following context to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error("❌ Error generating answer")
        st.text(traceback.format_exc())
        return None

# 📄 Process PDF
if uploaded_file and st.session_state.chunks is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("📖 Reading and embedding the PDF..."):
        text = extract_text_from_pdf(tmp_path)
        if text:
            st.session_state.chunks = split_text_into_chunks(text)
            try:
                embeddings = embedder.encode(st.session_state.chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))
                st.session_state.index = index
                st.success("✅ PDF processed and ready!")
            except Exception as e:
                st.error("❌ Error during embedding/indexing.")
                st.text(traceback.format_exc())

# 💬 Chat Interface
if st.session_state.chunks and st.session_state.index:
    st.markdown("---")
    st.subheader("💬 Chat with your PDF")

    # Display previous chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.markdown("---")

    # Input for new question
    new_question = st.text_input("Ask a new question (type 'exit' to end):", key="user_input")

    if new_question:
        if new_question.lower() == "exit":
            st.markdown("👋 Conversation ended. Thanks for chatting!")
        else:
            with st.spinner("💭 Thinking..."):
                answer = rag_qa(new_question, st.session_state.chunks, st.session_state.index)
                if answer:
                    st.session_state.chat_history.append((new_question, answer))
                    st.experimental_rerun()
