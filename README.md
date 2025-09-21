# Legal Bee - Bangladeshi Law Assistant

Legal Bee is a Retrieval-Augmented Generation (RAG) agent that provides legal support to Bangladeshi people according to Bangladeshi laws. It uses LangChain, Pinecone, Google's Gemini, and HuggingFace embedding models to provide accurate legal information in both Bangla and English.

## Features

- **Multilingual Support**: Handles queries in both Bangla and English
- **Input Classification**: Automatically classifies user input as legal questions, story analysis, or general conversation
- **Context-Aware Responses**: Retrieves relevant legal information from a vector database of Bangladeshi laws
- **Source Citations**: Provides citations to the relevant legal documents
- **Web Interface**: Simple and intuitive web interface for interacting with the agent

## Project Architecture

```
[User Input]
       ↓
[LangChain Router]
       ↓
[Input Classifier]
     ↙     ↘
[Story Analysis]  [Direct Question Handler]
       ↓              ↓
[Embedding Search → Vector Store (Pinecone)]
       ↓
[Context + Prompt + Tools]
       ↓
[LLM Response Generator]
       ↓
[Multilingual Output (Bangla/English)]
```

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r req.txt
   ```

2. Set up environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

3. Ingest PDF documents:
   ```
   python scripts/create_index.py
   python scripts/ingest_pdfs.py
   ```

4. Run the application:
   ```
   python main.py
   ```

5. Access the web interface at `http://localhost:8000`

## Components

- **PDF Ingestion**: Processes PDF documents containing Bangladeshi laws
- **Vector Database**: Stores document embeddings for efficient retrieval
- **LangChain Router**: Routes queries to appropriate handlers based on classification
- **Input Classifier**: Classifies user input into different categories
- **Response Generator**: Generates contextually relevant responses with citations
- **FastAPI Backend**: Provides API endpoints and serves the web interface

## Usage

1. Enter your legal question in either Bangla or English
2. The system will automatically detect the language and classify your query
3. Relevant legal information will be retrieved from the vector database
4. A response will be generated with citations to the relevant legal documents

## License

This project is licensed under the MIT License.