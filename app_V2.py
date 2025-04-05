from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import requests
import json
import PyPDF2
import re
from typing import List, Dict, Any, Optional

# Download NLTK data for sentence tokenization - fixed punkt download
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  # Download punkt instead of punkt_tab

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("Groq API key not found in .env file.")

app = FastAPI()

document_store = {}
embedding_store = {}
chunks_store = {}
conversation_history = {}
conversation_summaries = {}  # Store for conversation summaries

# Models initialization
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
DIMENSION = 384

class EmbedRequest(BaseModel):
    document: str
    chunk_size: int = 512
    chunk_overlap: int = 128
    semantic_chunking: bool = True

class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None
    top_k: int = 5
    rerank: bool = True

class ConversationRequest(BaseModel):
    conversation_id: str

def semantic_chunking(text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into semantic chunks based on sentence boundaries
    """
    # Handle empty or very short text
    if not text or len(text) < min_chunk_size:
        return [text] if text else []
        
    # First tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"Error in sentence tokenization: {e}")
        # Fallback to simple splitting by periods if NLTK fails
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size and we have content already,
        # save the current chunk and start a new one
        if len(current_chunk) + len(sentence) > max_chunk_size and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def sliding_window_chunking(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Split text into chunks using a sliding window approach
    """
    # Handle empty or very short text
    if not text or len(text) < chunk_size:
        return [text] if text else []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Get the chunk
        chunks.append(text[start:end])
        # Move start position with overlap consideration
        start = start + chunk_size - overlap
        
    return chunks

def process_document(file_path: str) -> str:
    text = ""
    if file_path.endswith('.pdf'):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Check if extraction was successful
                        text += page_text + " "
                    else:
                        print(f"Warning: Could not extract text from a page in {file_path}")
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    elif file_path.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(file_path)
            text = " ".join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ImportError("DOCX library not found. Please install python-docx: pip install python-docx")
        except Exception as e:
            print(f"Error processing DOCX: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing DOCX: {str(e)}")
            
    elif file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Error processing TXT: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing TXT: {str(e)}")
            
    # Clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_conversation(conversation_id: str) -> str:
    """
    Summarize a conversation history to maintain context while reducing token usage
    """
    if conversation_id not in conversation_history or len(conversation_history[conversation_id]) <= 2:
        return ""
    
    # Get last summary if it exists
    last_summary = conversation_summaries.get(conversation_id, "")
    
    # Get the last few interactions (e.g., last 3)
    recent_interactions = conversation_history[conversation_id][-3:]
    
    # Format recent interactions for summarization
    recent_text = ""
    for interaction in recent_interactions:
        recent_text += f"Q: {interaction['question']}\nA: {interaction['answer']}\n\n"
    
    # Create prompt for summarization
    prompt = f"""
    Previous conversation summary:
    {last_summary}
    
    Recent conversation:
    {recent_text}
    
    Please provide a concise summary of this conversation that captures the key points discussed so far.
    Keep the summary focused and brief (1-3 sentences).
    """
    
    # Generate summary
    summary = generate_response(prompt)
    
    # Store the new summary
    conversation_summaries[conversation_id] = summary
    
    return summary

@app.get("/")
async def root():
    return {"message": "Enhanced Document Intelligence RAG API is running!"}

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "documents_stored": len(document_store),
        "active_conversations": len(conversation_history)
    }

@app.post("/api/embedding")
async def embed_document(req: EmbedRequest):
    try:
        if not os.path.exists(req.document):
            raise HTTPException(status_code=400, detail=f"File not found: {req.document}")
            
        print(f"Processing document: {req.document}")
        text = process_document(req.document)
        
        if not text or len(text) < 10:  # Sanity check for empty documents
            raise HTTPException(status_code=400, detail="Document appears to be empty or could not be processed")
        
        print(f"Document processed, length: {len(text)} characters")
        
        # Choose chunking strategy based on request
        if req.semantic_chunking:
            print("Using semantic chunking")
            chunks = semantic_chunking(text, min_chunk_size=200, max_chunk_size=req.chunk_size)
        else:
            print("Using sliding window chunking")
            chunks = sliding_window_chunking(text, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
            
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to generate chunks from document")
            
        print(f"Document split into {len(chunks)} chunks")
        
        # Get embeddings for chunks
        print("Generating embeddings...")
        embeddings = embedding_model.encode(chunks)
        print(f"Generated {len(embeddings)} embeddings")
            
        # Set up HNSW index for efficient similarity search
        print("Setting up FAISS index")
        index = faiss.IndexHNSWFlat(DIMENSION, 32)  # 32 is the number of connections per layer
        index.hnsw.efConstruction = 40  # Controls index build quality (higher = better, slower)
        faiss_embeddings = np.array(embeddings).astype('float32')
        index.add(faiss_embeddings)
            
        doc_id = str(uuid.uuid4())
        document_store[doc_id] = index
        chunks_store[doc_id] = chunks
        embedding_store[doc_id] = embeddings
        
        print(f"Document embedded successfully with ID: {doc_id}")
        
        return {
            "message": "Document embedded successfully", 
            "document_id": doc_id, 
            "chunk_count": len(chunks),
            "chunking_method": "semantic" if req.semantic_chunking else "sliding_window"
        }
    except Exception as e:
        print(f"Error in embed_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed document: {str(e)}")

@app.post("/api/query")
async def query_document(req: QueryRequest):
    try:
        # Handle conversation ID
        if not req.conversation_id:
            conv_id = str(uuid.uuid4())
            conversation_history[conv_id] = []
        else:
            conv_id = req.conversation_id
            if conv_id not in conversation_history:
                conversation_history[conv_id] = []
        
        # Check if documents are embedded
        if not document_store:
            raise HTTPException(status_code=404, detail="No documents have been embedded yet")
        
        # Select document ID
        if req.document_id and req.document_id in document_store:
            doc_id = req.document_id
        elif len(document_store) > 0:
            doc_id = next(iter(document_store))
        else:
            raise HTTPException(status_code=404, detail="No documents available")
        
        # Generate embedding for the user query
        query_embed = embedding_model.encode([req.query])
        
        index = document_store[doc_id]

        # Configure search parameters
        if isinstance(index, faiss.IndexHNSWFlat):
            index.hnsw.efSearch = 64  # Higher = better accuracy, slower
        
        # Retrieve top-k similar chunks (we'll get more than needed for reranking)
        k = min(req.top_k * 2 if req.rerank else req.top_k, len(chunks_store[doc_id]))  # Get extra for reranking
        distances, indices = index.search(np.array(query_embed).astype('float32'), k)
        
        # Gather candidate chunks
        chunks = chunks_store[doc_id]
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                candidates.append({
                    "chunk": chunks[idx],
                    "score": float(distances[0][i]),
                    "index": int(idx)
                })
        
        # Apply reranking if requested
        if req.rerank and len(candidates) > 0:
            pairs = [(req.query, candidate["chunk"]) for candidate in candidates]
            rerank_scores = reranker_model.predict(pairs)
            
            # Sort candidates by reranker score
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            candidates = candidates[:req.top_k]  # Keep only top k after reranking
        
        # Build context from selected chunks
        context = ""
        for candidate in candidates[:req.top_k]:
            context += candidate["chunk"] + "\n\n"
        
        # Get conversation summary for memory
        conversation_summary = summarize_conversation(conv_id) if len(conversation_history[conv_id]) > 0 else ""
        
        # Build conversation history string (recent interactions)
        history = ""
        recent_interactions = conversation_history[conv_id][-3:] if len(conversation_history[conv_id]) > 0 else []
        for msg in recent_interactions:
            if 'question' in msg and 'answer' in msg:
                history += f"Q: {msg['question']}\nA: {msg['answer']}\n\n"
        
        # Prepare prompt for Groq API
        prompt = f"""
        Answer this question based on the context:

        CONTEXT:
        {context}

        CONVERSATION SUMMARY:
        {conversation_summary}
        
        RECENT CONVERSATION:
        {history}

        QUESTION: {req.query}

        If you can't find an answer in the context, say "I don't have enough information to answer this question."
        """
        
        # Generate response using Groq
        response_text = generate_response(prompt)
        
        # Save this interaction to history
        conversation_history[conv_id].append({"question": req.query, "answer": response_text})
        
        # Every 5 interactions, update the conversation summary
        if len(conversation_history[conv_id]) % 5 == 0:
            summarize_conversation(conv_id)
        
        return {
            "response": response_text,
            "conversation_id": conv_id,
            "document_id": doc_id,
            "top_chunks_used": len(candidates[:req.top_k]),
            "reranked": req.rerank
        }
    except Exception as e:
        print(f"Error in query_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/conversation/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    summary = conversation_summaries.get(conversation_id, "No summary available")
    
    return {
        "conversation_id": conversation_id,
        "interactions_count": len(conversation_history[conversation_id]),
        "summary": summary
    }

@app.post("/api/conversation/clear")
async def clear_conversation(req: ConversationRequest):
    if req.conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_history[req.conversation_id] = []
    if req.conversation_id in conversation_summaries:
        del conversation_summaries[req.conversation_id]
    
    return {"message": "Conversation history cleared", "conversation_id": req.conversation_id}

def generate_response(query: str) -> str:
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.3  # Lower temperature for more focused answers
        }
            
        response = requests.post(url, json=payload, headers=headers)
            
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            error_message = f"Error from Groq API (Status {response.status_code}): {response.text}"
            print(error_message) 
            return "I encountered an error when generating a response. Please try again."
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message) 
        return "An unexpected error occurred while processing your request."

# Optional endpoint for health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "embedding_model": "all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        "api_version": "1.1.0"
    }


try:
    nltk.download('punkt')
    print("NLTK punkt dataset downloaded successfully")
except Exception as e:
    print(f"Warning: Failed to download NLTK punkt dataset: {e}")
    
