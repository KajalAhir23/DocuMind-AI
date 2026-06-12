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

# Grab Hugging Face Token safely from Streamlit Secrets or local .env
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Page Configuration
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
st.title("📄 DocuMind AI")
st.markdown("### AI-Powered PDF Question Answering System")

# Validation check for API Token
if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Add HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ====================================================================
# 2. CACHED RAG INITIALIZATION (Prevents DB rebuild on rerun)
# ====================================================================
@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_bytes):
    # Internal imports to isolate initialization dependencies
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # Streamlit passes files as bytes. Write to a safe, temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    # Step 1: Extract Document Contents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Chunk Document Elements
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Step 3: Embed Text Vectors (CPU-optimized for Cloud instances)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 4: Construct ChromaDB Vector Index
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 5: Connect Remote Hugging Face Model Endpoint
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        task="text2text-generation",
        temperature=0.3,
        max_new_tokens=256,
        huggingfacehub_api_token=HF_TOKEN
    )

    # Step 6: Construct LCEL RAG Pipeline Chain
    template = """Use the context to answer the question.
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
    
    # Remove the temporary local system path safely
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
        # Cache uses file bytes to identify if a new file is uploaded
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
