# 📄 DocuMind AI — Document Q&A Chatbot

DocuMind AI is a production-ready, interactive Retrieval-Augmented Generation (RAG) web application that enables users to upload PDF, DOCX, and TXT files and ask questions about them. 

The chatbot retrieves the most contextually relevant chunks from the uploaded documents using semantic similarity search and synthesizes factually accurate answers using the high-performance **Groq API** with the **Llama 3 8B (`llama3-8b-8192`)** LLM. It is designed to never hallucinate, referencing exact citations (filename, page numbers, and text snippets) for every answer.

This project is ideal for **Resume Portfolios**, **Placement Preparation**, **Internship Applications**, and demonstrating end-to-end LLM application design.

---

## 🌟 Features

* **Multi-Format File Upload**: Supported file formats: PDF, DOCX (Microsoft Word), and TXT.
* **Batch Processing**: Upload single or multiple files concurrently.
* **Document Processing & Chunking**: Automatic character extraction, structure validation, and text segmenting using recursive separation.
* **Isolated Vector Database**: In-memory storage powered by **ChromaDB** with session-level separation to guarantee multi-tenant data privacy.
* **Context-Bound Chat**: Answers are generated strictly using the uploaded context. If the answer isn't in the documents, the system replies with: *"The answer could not be found in the uploaded document."*
* **Detailed Citation Footprints**: Shows document filename, page number (if available), and text snippets below every response.
* **Chat Session Management**: Keep track of conversation flow with a "Clear Chat" interface to completely wipe history and flush index cache.
* **Polished User Experience**: Interactive loading indicators, error modals, and responsive UI feedback for a premium experience.

---

## 🛠️ Technology Stack

* **Frontend UI**: [Streamlit](https://streamlit.io/) (v1.45.1)
* **LLM Orchestration**: [LangChain](https://www.langchain.com/) (v0.3.25)
* **LLM Engine**: [Groq API](https://groq.com/) with **Llama 3 8B (`llama3-8b-8192`)**
* **Embedding Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (v4.1.0)
* **Vector Store**: [ChromaDB](https://www.trychroma.com/) (v1.0.10)
* **File Parsers**: `pypdf` (v5.5.0) and `python-docx` (v1.1.2)
* **Runtime**: Python 3.11

---

## 📂 Project Structure

```text
DocuMind-AI/
├── app.py                  # Streamlit Web Application entry point
├── requirements.txt        # Python dependency manifest (exact versions)
├── README.md               # Project documentation
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
└── utils/
    ├── loaders.py          # File readers, text extraction & validations
    ├── embeddings.py       # Embedding model lazy initialization
    └── rag_pipeline.py     # Chunking, vector indexing & RetrievalQA chain
```

---

## ⚙️ Environment Setup

1. **Prerequisites**: Ensure you have Python 3.11 installed. You will also need a Groq Access API Key. Get your key from the [Groq Console](https://console.groq.com/keys).
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/DocuMind-AI.git
   cd DocuMind-AI
   ```
3. **Set Up the `.env` Configuration**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 🚀 Running Locally

1. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
   The application will automatically spin up in your default web browser at `http://localhost:8501`.

---

## 🌐 Deployment Steps

### Deploying to Streamlit Community Cloud

DocuMind AI is fully compatible with Streamlit Community Cloud:

1. Push your code to a public/private GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select your repository, branch, and set the entry file to `app.py`.
4. Expand **Advanced settings...** and add your Groq API Key under **Secrets**:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
5. Click **Deploy!**

---

## 📸 Interface Preview (Screenshots)

*(Add screenshots here showing the Document Upload, the Processing Success Alert, and the Chat Interface with Retrieved Sources)*

---

## 🔮 Future Improvements

* **Optical Character Recognition (OCR)**: Integrate `pytesseract` or `easyocr` to support scanned PDFs or image attachments.
* **Hybrid Search**: Combine BM25 keyword matching with Chroma semantic search to improve retrieval of highly specific codes or terms.
* **Response Reranking**: Incorporate Cohere or Hugging Face Cross-Encoders to rerank the top-retrieved documents for richer text synthesis.
* **Advanced Document Formatting**: Support parsing markdown, json, and CSV files directly.
