import streamlit as st
from app import (
    auto_extract_filters_from_query,
    build_raw_text_map,
    parse_year_metrics,
    extract_chunks_from_pdf,
    create_faiss_index,
    answer_question,
    get_top_chunks  # ✅ Make sure it's imported
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

            # ✅ Corrected call to get_top_chunks
            context = get_top_chunks(query, faiss_index, chunk_texts, embeddings, top_k=20)

            # ✅ If get_top_chunks returned empty context
            if not context.strip():
                response = "⚠️ I couldn't find enough context to answer that question."
            else:
                response = answer_question(query, faiss_index, chunk_texts, embeddings, year_metrics)

        except Exception as e:
            response = f"❌ Error: {e}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
