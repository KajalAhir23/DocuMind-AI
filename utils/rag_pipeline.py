import os
import uuid
import logging
from typing import List, Dict, Any
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from utils.embeddings import get_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_documents(documents: List[Document]) -> Chroma:
    """
    Splits input documents into chunks, generates embeddings, and indexes them
    in an isolated, ephemeral ChromaDB instance with a unique collection name.
    """
    logger.info(f"Splitting {len(documents)} document pages/paragraphs...")
    
    # Text Chunking strategy: 800 char size, 150 overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} text chunks.")
    
    # Embedding generation & Vector DB storage
    try:
        embeddings = get_embeddings()
        
        # Ephemeral Client for session-level isolation
        client = chromadb.EphemeralClient()
        
        # Generate a unique UUID-based collection name for session isolation
        unique_collection_name = f"documind_qa_{uuid.uuid4().hex}"
        logger.info(f"Building Chroma vector database with collection: {unique_collection_name}")
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=unique_collection_name
        )
        logger.info("Chroma vector database built successfully.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        raise RuntimeError(f"Failed to process documents and build vector database: {str(e)}")

def build_qa_chain(vector_store: Chroma, api_token: str, model_name: str = None):
    """
    Creates a modern LangChain retrieval chain using the 'ChatGroq' model.
    Uses only stable, non-deprecated APIs.
    """
    if not api_token or api_token.strip() == "":
        raise ValueError(
            "Groq API Key is empty or missing. "
            "Please provide a valid key in the sidebar or env file."
        )
        
    if not model_name or model_name.strip() == "":
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        
    logger.info(f"Initializing ChatGroq LLM with model: {model_name}...")
    try:
        llm = ChatGroq(
            model=model_name,
            groq_api_key=api_token.strip(),
            temperature=0.3
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        err_msg = str(e)
        if "decommissioned" in err_msg.lower() or "not found" in err_msg.lower() or "does not exist" in err_msg.lower():
            raise RuntimeError(
                f"Groq Model Error: The configured model '{model_name}' is decommissioned or invalid. "
                "Please configure a valid, active GROQ_MODEL in your environment."
            )
        raise RuntimeError(
            f"Failed to connect to Groq Inference API: {str(e)}"
        )

    # Prompt forcing model to stick strictly to the context and output specific message when not found
    system_prompt = (
        "Use the following pieces of context to answer the question at the end.\n"
        "If the answer cannot be found in the provided context, or if you are unsure, "
        "respond EXACTLY with:\n"
        "\"The answer could not be found in the uploaded document.\"\n"
        "Do not attempt to make up an answer, assume details, or utilize outside knowledge.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Top-K retrieval setting optimized to k=5 for better context coverage
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Modern LangChain 0.3 document and retrieval chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def query_rag_pipeline(qa_chain, question: str) -> Dict[str, Any]:
    """
    Executes a query against the retrieval chain and formats the response.
    Returns a dictionary containing 'answer' and a list of 'sources'.
    Performs robust post-processing validation to eliminate hallucinations.
    """
    logger.info(f"Querying pipeline with question: '{question}'")
    try:
        # Invoke using modern input key 'input'
        response = qa_chain.invoke({"input": question})
        
        answer = response.get("answer", "").strip()
        source_documents = response.get("context", [])
        
        # Post-processing to enforce strict context boundaries and prevent hallucinations
        cleaned_answer = answer.lower().strip()
        refusal_signatures = [
            "could not be found in the uploaded document",
            "not present in the context",
            "no information",
            "not mentioned",
            "does not mention",
            "does not contain",
            "not found in the document",
            "insufficient information"
        ]
        
        # If the answer is empty or matches any refusal phrases, override with standard polite message
        if not answer or any(sig in cleaned_answer for sig in refusal_signatures):
            answer = "The answer could not be found in the uploaded document."
            
        sources = []
        for doc in source_documents:
            source_name = doc.metadata.get("source", "Unknown Document")
            page_num = doc.metadata.get("page", "N/A")
            snippet = doc.page_content.strip()
            
            # Format snippet length to keep UI neat
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
                
            sources.append({
                "source": source_name,
                "page": page_num,
                "snippet": snippet
            })
            
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error querying RAG pipeline: {e}")
        err_msg = str(e)
        
        # Intercept specific API errors for Groq
        if "Authentication" in err_msg or "unauthorized" in err_msg.lower() or "401" in err_msg or "invalid api key" in err_msg.lower():
            raise ValueError(
                "Groq Authentication Failed. Please verify your API Key credentials in the sidebar."
            )
        elif "decommissioned" in err_msg.lower() or "not found" in err_msg.lower() or "does not exist" in err_msg.lower() or ("model" in err_msg.lower() and "support" in err_msg.lower()):
            raise RuntimeError(
                "The Groq model configured is decommissioned or not supported. "
                "Please configure a valid, active GROQ_MODEL (e.g. llama-3.3-70b-versatile) in your environment settings."
            )
        elif "rate limit" in err_msg.lower() or "429" in err_msg:
            raise RuntimeError(
                "Groq API Rate Limit exceeded. Please pause a moment before resubmitting."
            )
        elif "connection" in err_msg.lower() or "timeout" in err_msg.lower() or "503" in err_msg:
            raise RuntimeError(
                "Network timeout or server connectivity issue. Please check your internet connection and try again."
            )
        else:
            raise RuntimeError(
                f"Failed to query the document: {err_msg}"
            )
