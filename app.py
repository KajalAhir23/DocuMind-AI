# ====================================================================
# 1. THE SQLITE FIX (MUST BE AT THE ABSOLUTE TOP BEFORE ANY OTHER IMPORTS)
# ====================================================================
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Grab Gemini API Key safely from Streamlit Secrets or local .env
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Page Configuration
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
st.title("📄 DocuMind AI")
st.markdown("### AI-Powered PDF Question Answering System (Gemini Edition)")

# Validation check for API Token
if not GOOGLE_API_KEY:
    st.error("❌ Google Gemini API key not found. Add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

# ====================================================================
# 2. CACHED RAG INITIALIZATION (Prevents DB rebuild on rerun)
# ====================================================================
@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_bytes):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Write file bytes to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    # Step 1: Extract Document Contents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Chunk Document Elements
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Step 3: Embed Text Vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 4: Construct ChromaDB Vector Index
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 5: Connect Gemini LLM
    # Step 5: Connect Gemini LLM (Configured to forcefully bypass the broken v1beta endpoint)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        max_output_tokens=512,
        client_options={"api_version": "v1"}
    )
    # Step 6: Construct LCEL RAG Pipeline Chain
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        os.remove(pdf_path)
    except Exception:
        pass
        
    return chain

# ====================================================================
# 3. APPLICATION UI LOGIC INTERFACE
# ====================================================================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        
        with st.spinner("🧠 Initializing AI Engine & Processing PDF (This runs once)..."):
            chain = initialize_rag(file_bytes)
        st.success("✅ Document Processing Complete!")

        st.divider()
        question = st.text_input("💬 Ask a question about the PDF:")

        if question:
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.subheader("✅ Answer")
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
