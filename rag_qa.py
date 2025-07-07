# prompt: convert this notebook data into script

import sys
import os
from google.colab import drive
from google.colab import files
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# Check if running in Colab to handle file uploads and drive mounting
if 'google.colab' in sys.modules:
    drive.mount('/content/drive')
    # Install libraries if not already installed
    !pip install -q PyMuPDF faiss-cpu google-generativeai sentence-transformers

    print("📤 Upload your PDF file:")
    uploaded = files.upload()
    pdf_path = list(uploaded.keys())[0]
else:
    # For running as a standalone script, prompt for file path
    pdf_path = input("Enter the path to your PDF file: ")
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        sys.exit(1)


def extract_text_from_pdf(path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

pdf_text = extract_text_from_pdf(pdf_path)
if not pdf_text:
    print("Exiting due to error in PDF processing.")
    sys.exit(1)
print("✅ Text extracted from PDF!")

def split_text_into_chunks(text, chunk_size=300):
    """Splits text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = split_text_into_chunks(pdf_text)

# 🧠 Step 4: Embed chunks
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks)
except Exception as e:
    print(f"Error during embedding: {e}")
    sys.exit(1)

# 📚 Step 5: Store embeddings in FAISS
try:
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    print("✅ Embeddings indexed with FAISS")
except Exception as e:
    print(f"Error indexing embeddings with FAISS: {e}")
    sys.exit(1)

# 🤖 Step 6: Setup Gemini Flash model
# IMPORTANT: Replace "YOUR_API_KEY" with your actual API key.
# Consider using environment variables for sensitive information.
try:
    genai.configure(api_key="AIzaSyB2mnl5llLuKKhNbehBHmumfYu03cqr9VU")
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
except Exception as e:
    print(f"Error configuring or loading Gemini model: {e}")
    sys.exit(1)


# 💬 Step 7: Define RAG QA function
def rag_qa(query, chunks, index, embeddings, top_k=3):
    """Performs RAG-based question answering."""
    try:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        context = "\n".join([chunks[i] for i in indices[0]])
        prompt = f"Answer the question using the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during RAG QA: {e}")
        return "Could not generate an answer."

# 🚀 Step 8: Ask Questions!
print("\n🔍 You can now ask questions about the PDF! Type 'exit' to stop.")
while True:
    user_input = input("\n❓ Your question: ")
    if user_input.lower() == 'exit':
        print("👋 Thank you for using our services...")
        break
    answer = rag_qa(user_input, chunks, index, embeddings)
    print("\n🧠 Answer:", answer)
