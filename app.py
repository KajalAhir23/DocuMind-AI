import os
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv

# Load default environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up page configurations
st.set_page_config(
    page_title="DocuMind AI — Document Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS injection
st.markdown(
    """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global styling overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card design */
    .header-card {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2), 0 4px 6px -4px rgba(99, 102, 241, 0.2);
        margin-bottom: 25px;
        color: white;
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        font-weight: 300;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling overrides */
    .css-1644z7q, [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Chat message container styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 15px;
        padding: 10px 15px;
    }
    
    /* Sources snippet box */
    .source-box {
        background-color: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    
    .source-meta {
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from utils.loaders import load_document
from utils.rag_pipeline import process_documents, build_qa_chain, query_rag_pipeline

# Initialize session state variables securely
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = [] # list of tuples: (filename, size)

# Helper function to clear application states
def clear_all_states():
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.processed_files = []

# ----------------- SIDEBAR SECTION -----------------
with st.sidebar:
    st.markdown("### 📄 About Project")
    st.markdown(
        """
        **DocuMind AI** is a professional Retrieval-Augmented Generation (RAG) tool designed for recruiters, students, and engineers.
        
        It lets you upload multiple documents (PDFs, Word Docs, or Text logs) and queries them directly to give factual, context-aware answers.
        """
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Groq API Configuration")
    # Fetch environment API key securely
    env_token = os.getenv("GROQ_API_KEY", "")
    api_token = st.text_input(
        "Groq API Key:",
        value=st.session_state.get("groq_key", env_token),
        type="password",
        help="Paste your Groq API key here. Obtain it from console.groq.com/keys"
    )
    # Store token securely in session state
    st.session_state.groq_key = api_token
    
    # Fetch environment model name securely with safe fallback
    env_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    if not env_model:
        env_model = "llama-3.3-70b-versatile"
    
    model_name = st.text_input(
        "Groq Model ID:",
        value=st.session_state.get("groq_model", env_model),
        help="Specify the Groq model ID (e.g., llama-3.3-70b-versatile, llama-3.1-8b-instant)"
    )
    # Store model name securely in session state
    st.session_state.groq_model = model_name
    
    if not api_token:
        st.warning("⚠️ API Key missing. Configure it above or in the .env file.")
    else:
        st.success("🔑 API Key configured.")
        
    st.markdown("---")
    
    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        f"""
        * **Frontend:** Streamlit 1.45
        * **Orchestration:** LangChain 0.3 (Modern Chains)
        * **Vector Database:** ChromaDB 1.0 (UUID Isolated)
        * **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
        * **LLM Engine:** Groq API (`{model_name}`)
        * **Document Parsers:** pypdf & python-docx
        """
    )
    
    st.markdown("---")
    
    st.markdown("### 📝 Instructions")
    st.markdown(
        """
        1. Input your Groq API key.
        2. Upload your documents in the main window.
        3. Click **Process Documents** to generate embeddings.
        4. Ask questions using the chat input at the bottom.
        5. Review citations/sources below each response.
        """
    )
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat & Reset Index", use_container_width=True):
        clear_all_states()
        st.toast("Chat history and search index cleared!")
        st.rerun()

# ----------------- MAIN APP HEADER -----------------
st.markdown(
    f"""
    <div class="header-card">
        <h1 class="header-title">📄 DocuMind AI</h1>
        <p class="header-subtitle">Document Q&A Chatbot powered by LangChain, ChromaDB & Groq {model_name}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- DOCUMENT UPLOAD -----------------
st.markdown("### 📥 1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT documents (Single or Multiple):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Compute fingerprint of uploaded files to automatically detect changes
current_files_fingerprint = [(f.name, f.size) for f in uploaded_files] if uploaded_files else []
files_changed = current_files_fingerprint != st.session_state.processed_files

if uploaded_files:
    if files_changed:
        st.warning("⚠️ Document changes detected! Please click 'Process Documents' to initialize/update the index.")
    
    # Process button
    if st.button("⚡ Process Documents", type="primary", use_container_width=True):
        if not api_token:
            st.error("Error: Please provide a Groq API Key in the sidebar before processing documents.")
        else:
            try:
                all_documents = []
                failed_files = []
                
                # Step 1: Loading
                with st.spinner("🔄 Loading and extracting text from documents..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            try:
                                # Write uploaded file to temp disk path
                                with open(temp_path, "wb") as temp_f:
                                    temp_f.write(uploaded_file.getbuffer())
                                
                                # Parse document using custom loaders
                                docs = load_document(temp_path)
                                all_documents.extend(docs)
                            except Exception as file_err:
                                failed_files.append((uploaded_file.name, str(file_err)))
                                logger.error(f"Failed to process '{uploaded_file.name}': {file_err}")
                
                # Present non-blocking warnings for any file extraction failures
                if failed_files:
                    for fname, err in failed_files:
                        st.warning(f"⚠️ **Could not process '{fname}'**: {err}")
                
                if not all_documents:
                    st.error("Could not extract any text from the uploaded files. Please check the file formatting.")
                else:
                    # Step 2 & 3: Embeddings & Vector DB
                    with st.spinner("🧠 Generating embeddings & building vector database..."):
                        vector_store = process_documents(all_documents)
                        st.session_state.vector_store = vector_store
                    
                    # Step 4: QA Chain
                    with st.spinner("🤖 Connecting to Groq Inference API..."):
                        qa_chain = build_qa_chain(vector_store, api_token, model_name=model_name)
                        st.session_state.qa_chain = qa_chain
                        
                    # Save fingerprint to state
                    st.session_state.processed_files = current_files_fingerprint
                    st.success(f"Successfully processed {len(uploaded_files) - len(failed_files)} file(s)! {len(all_documents)} segment(s) indexed.")
                    st.toast("Knowledge base updated successfully!")
            except Exception as pipeline_err:
                logger.error(f"RAG Compilation Error: {pipeline_err}")
                st.error("An error occurred while compiling the RAG search index. Please verify your API key and try again.")
                st.session_state.vector_store = None
                st.session_state.qa_chain = None
else:
    # Clear index states automatically if all files are removed
    if st.session_state.processed_files:
        clear_all_states()
        st.rerun()

# ----------------- QA CHAT INTERFACE -----------------
st.markdown("---")
st.markdown("### 💬 2. Question Answering Chat")

# Inform the user if the index is ready
if not st.session_state.qa_chain:
    st.info("ℹ️ Upload and click 'Process Documents' to initialize the Chat Interface.")
else:
    # Display message history
    for msg in st.session_state.messages:
        role_avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=role_avatar):
            st.markdown(msg["content"])
            # Render sources if available for assistant messages
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("🔍 Retrieved Sources & Context", expanded=False):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-meta">#{idx + 1} - Source: {src['source']} | Page: {src['page']}</div>
                                <i>"{src['snippet']}"</i>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat input box
    if user_query := st.chat_input("Ask a question about the uploaded documents..."):
        # Append user query to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user query
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
            
        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    # Query RAG
                    result = query_rag_pipeline(st.session_state.qa_chain, user_query)
                    
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    # Display response
                    st.markdown(answer)
                    
                    # Render sources
                    if sources and answer != "The answer could not be found in the uploaded document.":
                        with st.expander("🔍 Retrieved Sources & Context", expanded=False):
                            for idx, src in enumerate(sources):
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <div class="source-meta">#{idx + 1} - Source: {src['source']} | Page: {src['page']}</div>
                                        <i>"{src['snippet']}"</i>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    
                    # Save response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except (ValueError, RuntimeError) as api_err:
                    # Handle known/parsed system and API validation errors cleanly
                    err_msg = str(api_err)
                    logger.warning(f"QA pipeline warning: {err_msg}")
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ **Service Notice**: {err_msg}",
                        "sources": []
                    })
                except Exception as unhandled_err:
                    # Secure unhandled errors by logging details internally and showing clean prompt
                    logger.exception("Unhandled error during QA generation:")
                    user_err_msg = "An unexpected error occurred while retrieving the answer. Please try again later."
                    st.error(user_err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ **Error**: {user_err_msg}",
                        "sources": []
                    })
