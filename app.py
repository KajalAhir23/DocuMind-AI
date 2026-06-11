from dotenv import load_dotenv
import os
load_dotenv()  

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile

# ── Page config ──────────────────────────────
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it!")

# ── Upload PDF ────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    # ── Step 1: Load PDF ──────────────────────
    st.info("Loading PDF...")
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # ── Step 2: Split into chunks ─────────────
    st.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    st.success(f"Created {len(chunks)} chunks from your PDF!")

    # ── Step 3: Generate Embeddings ───────────
    st.info("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Step 4: Store in ChromaDB ─────────────
    st.info("Storing in ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    st.success("Vector store ready!")

    # ── Step 5: Load LLM ──────────────────────
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # ── Step 6: Create RAG Chain ──────────────
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    # ── Step 7: Ask Questions ─────────────────
    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question here...")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.success("Answer:")
        st.write(answer)