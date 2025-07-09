# chatbot_ui.py

import streamlit as st
from app import auto_extract_filters_from_query, get_top_chunks, get_gemini_answer

st.set_page_config(page_title="🎓 Placement Chatbot", layout="centered")

st.title("📊 IGDTUW Placement Chatbot")
st.markdown("Ask me about placements — like top recruiters, highest CTC, or average offers per branch/year!")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
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
                response = "⚠️ No relevant data found. Please try a broader question."
            else:
                response = get_gemini_answer(query, top_chunks)

        except Exception as e:
            response = f"❌ Error: {e}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
