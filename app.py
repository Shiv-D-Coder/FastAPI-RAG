from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import json
import PyPDF2

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("Groq API key not found in .env file.")

app = FastAPI()

document_store = {}
embedding_store = {}
chunks_store = {}
conversation_history = {}
model = SentenceTransformer('all-MiniLM-L6-v2')
DIMENSION = 384

class EmbedRequest(BaseModel):
    document: str

class QueryRequest(BaseModel):
    query: str
    document_id: str = None  
    conversation_id: str = None

def process_document(file_path: str) -> str:
    text = ""
    if file_path.endswith('.pdf'):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + " "
        except ImportError:
            # Fallback to report error
            raise ImportError("PDF is not there.")
            
    elif file_path.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(file_path)
            text = " ".join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ImportError("DOCX library not found. Please install python-docx: pip install python-docx")
            
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
    return text

@app.get("/")
async def root():
    return {"message": "Document Intelligence RAG API is running!"}

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "documents_stored": len(document_store),
        "active_conversations": len(conversation_history)
    }

# @app.post("/api/embedding")
# async def embed_document(req: EmbedRequest):
#     try:
#         if not os.path.exists(req.document):
#             raise HTTPException(status_code=400, detail="File not found")
            
#         text = process_document(req.document)
#         chunks = [text[i:i + 512] for i in range(0, len(text), 512)]  # Simple chunking strategy
            
#         embeddings = model.encode(chunks)
            
#         index = faiss.IndexFlatL2(DIMENSION)
#         index.add(np.array(embeddings).astype('float32'))
            
#         doc_id = str(uuid.uuid4())
#         document_store[doc_id] = index
#         chunks_store[doc_id] = chunks
        
#         return {"message": "Document embedded successfully", "document_id": doc_id}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to embed document: {str(e)}")


@app.post("/api/embedding")
async def embed_document(req: EmbedRequest):
    try:
        if not os.path.exists(req.document):
            raise HTTPException(status_code=400, detail="File not found")
            
        text = process_document(req.document)
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]  # Simple chunking strategy
            
        embeddings = model.encode(chunks)
            
        index = faiss.IndexHNSWFlat(DIMENSION, 32)
        index.hnsw.efConstruction = 40
        index.add(np.array(embeddings).astype('float32'))
            
        doc_id = str(uuid.uuid4())
        document_store[doc_id] = index
        chunks_store[doc_id] = chunks
        
        return {"message": "Document embedded successfully", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed document: {str(e)}")


# @app.post("/api/query")
# async def query_document(req: QueryRequest):
    
    try:
        # conversation ID
        if not req.conversation_id:
            conv_id = str(uuid.uuid4())
            conversation_history[conv_id] = []
        else:
            conv_id = req.conversation_id
            if conv_id not in conversation_history:
                conversation_history[conv_id] = []
            
        if not document_store:
            raise HTTPException(status_code=404, detail="No documents have been embedded yet")
        
        if req.document_id and req.document_id in document_store:
            doc_id = req.document_id
        elif len(document_store) > 0:
            doc_id = next(iter(document_store))
        else:
            raise HTTPException(status_code=404, detail="No documents available")
            
        # query embedding
        query_embed = model.encode([req.query])
        
        index = document_store[doc_id]
        k = 3
        distances, indices = index.search(np.array(query_embed).astype('float32'), k)
        
        # relevant chunks
        chunks = chunks_store[doc_id]
        context = ""
        for idx in indices[0]:
            if idx < len(chunks):
                context += chunks[idx] + "\n\n"
        
        # conversation history
        history = ""
        if conv_id in conversation_history:
            for msg in conversation_history[conv_id]:
                if 'question' in msg and 'answer' in msg:
                    history += f"Q: {msg['question']}\nA: {msg['answer']}\n\n"
        
        # Create prompt
        prompt = f"""
        Answer this question based on the context:
        
        CONTEXT:
        {context}
        
        CONVERSATION HISTORY:
        {history}
        
        QUESTION: {req.query}
        
        If you can't find an answer in the context, say "I don't have enough information to answer this question."
        """
        
        # Generate response
        response_text = generate_response(prompt)
        
        # Update conversation history
        conversation_history[conv_id].append({"question": req.query, "answer": response_text})
        
        return {
            "response": response_text,
            "conversation_id": conv_id,
            "document_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

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
        query_embed = model.encode([req.query])
        
        index = document_store[doc_id]

        # Optional: Set efSearch parameter for HNSW index
        if isinstance(index, faiss.IndexHNSWFlat):
            index.hnsw.efSearch = 64  # Higher = better accuracy, slower
        
        # Perform the search (top-k similar chunks)
        k = 3
        distances, indices = index.search(np.array(query_embed).astype('float32'), k)
        
        # Gather relevant chunks
        chunks = chunks_store[doc_id]
        context = ""
        for idx in indices[0]:
            if idx < len(chunks):
                context += chunks[idx] + "\n\n"
        
        # Build conversation history string
        history = ""
        if conv_id in conversation_history:
            for msg in conversation_history[conv_id]:
                if 'question' in msg and 'answer' in msg:
                    history += f"Q: {msg['question']}\nA: {msg['answer']}\n\n"
        
        # Prepare prompt for Groq API
        prompt = f"""
        Answer this question based on the context:

        CONTEXT:
        {context}

        CONVERSATION HISTORY:
        {history}

        QUESTION: {req.query}

        If you can't find an answer in the context, say "I don't have enough information to answer this question."
        """
        
        # Generate response using Groq
        response_text = generate_response(prompt)
        
        # Save this interaction to history
        conversation_history[conv_id].append({"question": req.query, "answer": response_text})
        
        return {
            "response": response_text,
            "conversation_id": conv_id,
            "document_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


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
            ]
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

