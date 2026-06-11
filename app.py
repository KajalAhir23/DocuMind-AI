import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
st.title("📄 DocuMind AI")
st.markdown("### AI-Powered PDF Question Answering System")

if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Add HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("📖 Reading PDF..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        st.success(f"✅ Loaded {len(documents)} pages")

        with st.spinner("✂️ Splitting into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
        st.success(f"✅ Created {len(chunks)} chunks")

        with st.spinner("🧠 Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

        with st.spinner("💾 Building vector database..."):
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        st.success("✅ Vector Database Ready!")

        with st.spinner("🤖 Loading LLM..."):
            llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                task="text2text-generation",
                temperature=0.3,
                max_new_tokens=256,
                huggingfacehub_api_token=HF_TOKEN
            )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

        st.divider()
        question = st.text_input("💬 Ask a question about the PDF:")

        if question:
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.subheader("✅ Answer")
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
