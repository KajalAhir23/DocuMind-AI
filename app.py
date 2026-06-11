from dotenv import load_dotenv
import os
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA
import tempfile

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 DocuMind AI — PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    st.info("Loading PDF...")
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    st.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.success(f"Created {len(chunks)} chunks!")

    st.info("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.info("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    st.success("Vector store ready!")

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question here...")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.success("Answer:")
        st.write(answer)