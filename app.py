# ====================================================================
# 1. THE CRITICAL ENVIRONMENT & SQLITE FIXES (MUST BE AT THE ABSOLUTE TOP)
# ====================================================================
import os
import sys

# Force pure-Python implementation to completely bypass the protobuf version clash
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
st.title("📄 DocuMind AI")
st.markdown("### AI-Powered PDF Question Answering System (Gemini Edition)")

# Grab Gemini API Key safely from Streamlit Secrets or local environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    try:
        # st.secrets behaves like a dict but raises FileNotFoundError if secrets.toml is missing
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# Sidebar input fallback for API key if not set in environment or secrets
if not GOOGLE_API_KEY:
    st.sidebar.info("🔑 Google Gemini API Key required")
    api_key_input = st.sidebar.text_input(
        "Enter your Google Gemini API Key:", 
        type="password", 
        placeholder="AIzaSy..."
    )
    if api_key_input:
        GOOGLE_API_KEY = api_key_input
    else:
        st.warning("👈 Please enter your Google Gemini API Key in the sidebar to get started.")
        st.stop()

# ====================================================================
# 2. CACHED RAG INITIALIZATION (Prevents DB rebuild on rerun)
# ====================================================================
@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_bytes):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

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

    # Step 3: Embed Text Vectors (Using Cloud Google Embeddings to bypass local PyTorch c10.dll errors)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Step 4: Construct ChromaDB Vector Index
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 5: Connect Stable Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        max_output_tokens=512,
        api_version="v1"
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
