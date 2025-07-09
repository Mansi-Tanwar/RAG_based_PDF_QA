import streamlit as st
from app import (
    auto_extract_filters_from_query,
    get_top_chunks,
    get_gemini_answer,
    build_raw_text_map,
    parse_year_metrics,
    extract_chunks_from_pdf,
    create_faiss_index
)
import os
import re

# ✅ Step 1: Load & process PDFs from folder
pdf_folder_path = "./Placement Data"
raw_text_map = build_raw_text_map(pdf_folder_path)
year_metrics = parse_year_metrics(raw_text_map)

# ✅ Step 2: Extract chunks and build FAISS index
all_chunks = []
for fname in os.listdir(pdf_folder_path):
    if not fname.lower().endswith(".pdf"):
        continue
    match = re.search(r'(\d{4})', fname)
    year = match.group(1) if match else None
    if year:
        all_chunks.extend(extract_chunks_from_pdf(os.path.join(pdf_folder_path, fname), year))

faiss_index, chunk_texts, embeddings = create_faiss_index(all_chunks)

# ✅ Streamlit UI
st.set_page_config(page_title="🎓 Placement Chatbot", layout="centered")

st.title("📊 IGDTUW Placement Chatbot")
st.markdown("Ask me about placements — like top recruiters, highest CTC, or average offers per branch/year!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask your placement-related question...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        try:
            year_filter, dept_filter = auto_extract_filters_from_query(query)
            st.markdown(f"🔍 **Filters applied** → Year: `{year_filter}`, Department: `{dept_filter}`")

            top_chunks = get_top_chunks(query, year_filter=year_filter, dept_filter=dept_filter, k=20)

            if not top_chunks:
                context = "\n".join(chunk_texts[:20])  # fallback: give first 20 chunks
            else:
                context = "\n".join([chunk["text"] for chunk in top_chunks])

            response = get_gemini_answer(query, top_chunks) if top_chunks else "⚠️ No relevant chunks found."
        except Exception as e:
            response = f"❌ Error: {e}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
