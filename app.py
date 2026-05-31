# Phase 5 — Streamlit UI (app.py)

import streamlit as st
import tempfile, os
from rag_pipeline import load_and_split, build_vector_store, build_rag_chain, ask

st.title("📄 RAG Chatbot")
st.caption("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        chunks = load_and_split(tmp_path)
        db = build_vector_store(chunks)
        chain = build_rag_chain(db)
    st.success("Ready! Ask your questions below.")

    question = st.text_input("Ask something about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = ask(chain, question)
        st.markdown(f"**Answer:** {answer}")