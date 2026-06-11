# 📄 DocuMind AI — PDF Q&A Chatbot

A RAG-based (Retrieval Augmented Generation) intelligent chatbot that lets you upload any PDF and ask questions about its content using AI.

---

## 🚀 Live Demo
🔗 [Click here to try the app](https://kpkvulcskbewfqk7bdjymk.streamlit.app)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline & LLM chaining |
| ChromaDB | Vector database for storing embeddings |
| HuggingFace | Embeddings & LLM (flan-t5-base) |
| Streamlit | Web UI |
| PyPDF | PDF loading & parsing |

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding Generation
→ Store in ChromaDB → User Question → Similarity Search
→ Relevant Chunks → LLM → Answer
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/KajalAhir23/DocuMind-AI.git
cd DocuMind-AI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create `.env` file**
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

**4. Run the app**
```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
DocuMind-AI/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env                # API token (not pushed to GitHub)
├── .gitignore          # Git ignore rules
└── uploaded_pdfs/      # Temp PDF storage
```

---

## ✨ Features

- 📤 Upload any PDF file
- 🔪 Automatic text chunking
- 🧮 Embedding generation via HuggingFace API
- 🗄️ Vector storage using ChromaDB
- 🤖 AI-powered answers using google/flan-t5-base
- 💬 Simple and clean chat interface

---

## 👩‍💻 Developer

**Kajal Bhatiya**
- 🔗 [LinkedIn](https://www.linkedin.com/in/kajal-bhatiya-276721306/)
- 🐙 [GitHub](https://github.com/KajalAhir23)
- 💻 [LeetCode](https://leetcode.com/u/KajalBhatiya/)
