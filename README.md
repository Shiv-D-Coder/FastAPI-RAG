# Document Intelligence RAG Chatbot System

This project implements a Document Intelligence Retrieval-Augmented Generation (RAG) API using FastAPI. The API allows users to upload documents, embed them for semantic search, and query the embedded documents for contextual responses.

## Features

- Upload and embed documents in PDF, DOCX, and TXT formats.
- Query embedded documents to retrieve relevant information.
- Maintain conversation history for contextual responses.
- Integration with Groq API for generating answers based on document context.

## Requirements

- Python 3.11 or higher
- Required Python packages (see below)

## Installation

1. Clone the repository:

```
git clone https://github.com/Shiv-D-Coder/FastAPI-RAG
cd FastAPI-RAG

```

2. Create a virtual environment:

```
python -m venv myenv
myenv\Scripts\activate
```


3. Install the required packages:

```
pip install fastapi uvicorn sentence-transformers faiss-cpu python-dotenv requests PyPDF2 python-docx
```

text

4. Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```


## Running the Application

To run the FastAPI application, use the following command:

```
uvicorn main:app --reload
```


The API will be accessible at `http://127.0.0.1:8000`.

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns a message indicating that the API is running.


### 2. Status Endpoint

**GET** `/api/status`

Returns the current status of the API, including the number of stored documents and active conversations.

**Response**:
{
"status": "online",
"documents_stored": <number>,
"active_conversations": <number>
}



### 3. Embedding Endpoint

**POST** `/api/embedding`

Embeds a document for semantic search.

**Request Body**:
{
"document": "path/to/your/document.pdf"
}


### 4. Query Endpoint

**POST** `/api/query`

Queries the embedded documents for relevant information.

**Request Body**:
{
"query": "What is the main argument in the document?",
"conversation_id": null // Optional
}

