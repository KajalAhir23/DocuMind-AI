from dotenv import load_dotenv
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocuMind AI")
st.markdown("### AI-Powered PDF Question Answering System")

# -------------------------
# Check Token
# -------------------------
if not HF_TOKEN:
    st.error(
        "❌ Hugging Face token not found.\n\n"
        "Add HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets or .env file."
    )
    st.stop()

# -------------------------
# Session State
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Upload PDF
# -------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        # -------------------------
        # Load PDF
        # -------------------------
        with st.spinner("📖 Reading PDF..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

        st.success(f"Loaded {len(documents)} pages")

        # -------------------------
        # Split Text
        # -------------------------
        with st.spinner("✂️ Splitting text into chunks..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

        st.success(f"Created {len(chunks)} chunks")

        # -------------------------
        # Embeddings
        # -------------------------
        with st.spinner("🧠 Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

        # -------------------------
        # Vector Store
        # -------------------------
        with st.spinner("💾 Building vector database..."):
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

        st.success("✅ Vector Database Ready")

        # -------------------------
        # Hugging Face LLM
        # -------------------------
        with st.spinner("🤖 Loading LLM..."):
            from langchain_community.llms import HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=HF_TOKEN,
                model_kwargs={"temperature": 0.3, "max_new_tokens": 256}
            )

        # -------------------------
        # QA Chain
        # -------------------------
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )

        st.divider()

        # -------------------------
        # Ask Question
        # -------------------------
        question = st.text_input(
            "💬 Ask a question about the PDF:"
        )

        if question:

            with st.spinner("Thinking..."):

                result = qa_chain.invoke(
                    {"query": question}
                )

                answer = result["result"]

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

            st.subheader("✅ Answer")
            st.write(answer)

            # -------------------------
            # Sources
            # -------------------------
            if result.get("source_documents"):

                with st.expander("📚 Source References"):

                    for i, doc in enumerate(
                        result["source_documents"],
                        start=1
                    ):

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        st.markdown(
                            f"**Source {i} - Page {page + 1 if isinstance(page, int) else page}**"
                        )

                        st.write(
                            doc.page_content[:500] + "..."
                        )

        # -------------------------
        # Chat History
        # -------------------------
        if st.session_state.chat_history:

            st.divider()
            st.subheader("📝 Chat History")

            for idx, item in enumerate(
                reversed(st.session_state.chat_history),
                start=1
            ):
                st.markdown(
                    f"**Q{idx}:** {item['question']}"
                )
                st.write(
                    f"**Answer:** {item['answer']}"
                )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
