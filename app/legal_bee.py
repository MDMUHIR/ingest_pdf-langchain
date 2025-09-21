import os, re
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document

load_dotenv()

# HuggingFace embeddings (must match ingestion)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Gemini for answering
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Input classifier prompt
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an input classifier for a legal assistant. 
    Classify the user's input into one of these categories:
    - LEGAL_QUESTION: Direct questions about laws, legal procedures, or rights
    - STORY_ANALYSIS: Personal situations that need legal context
    - GENERAL: General conversation not requiring legal knowledge
    
    Output ONLY the category name, nothing else."""),
    ("user", "{input}")
])

classifier_chain = classifier_prompt | llm | StrOutputParser()

# Format documents for context
def format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        meta = d.metadata
        cite = f"{meta.get('title','Unknown')} (p.{meta.get('page','?')})"
        lines.append(f"[{cite}] {d.page_content}")
    return "\n\n".join(lines)

# Detect language (Bangla or English)
def detect_language(text: str) -> str:
    """Detect if text contains Bangla characters."""
    return "bangla" if re.search(r"[\u0980-\u09FF]", text) else "english"

# Legal question handler prompt
legal_question_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Legal Bee, a legal information assistant for Bangladeshi law.
    
    Rules:
    - Provide accurate information based ONLY on the legal context provided
    - Do NOT give personal legal advice, only provide information from the laws
    - Always cite sources like (Act title, page)
    - If unsure or if the context doesn't contain relevant information, say "I don't have enough information about this specific legal matter"
    - Answer in the same language as the question (Bangla if Bangla, English if English)
    - Be concise but thorough
    """),
    ("user", "Question: {input}\n\nContext:\n{context}\n\nAnswer with citations:")
])

# Story analysis prompt
story_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Legal Bee, a legal information assistant for Bangladeshi law.
    
    The user has shared a personal situation that may have legal implications. 
    
    Rules:
    - Analyze the situation and identify relevant legal concepts
    - Provide information based ONLY on the legal context provided
    - Do NOT give personal legal advice, only provide information from the laws
    - Always cite sources like (Act title, page)
    - If unsure or if the context doesn't contain relevant information, say "I don't have enough information about this specific legal matter"
    - Answer in the same language as the input (Bangla if Bangla, English if English)
    - Be empathetic but factual
    """),
    ("user", "Situation: {input}\n\nContext:\n{context}\n\nAnalysis with citations:")
])

# General conversation prompt
general_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Legal Bee, a friendly legal information assistant for Bangladeshi law.
    
    The user has asked a general question that doesn't require specific legal knowledge.
    
    Rules:
    - Be helpful and friendly
    - If they're asking for legal information, politely suggest they rephrase as a specific legal question
    - Answer in the same language as the input (Bangla if Bangla, English if English)
    """),
    ("user", "{input}")
])

# Define the branches based on classification
def create_router_chain():
    # Legal question handler chain
    legal_question_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input": (lambda x: x["input"])
        }
        | legal_question_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Story analysis chain
    story_analysis_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input": (lambda x: x["input"])
        }
        | story_analysis_prompt 
        | llm 
        | StrOutputParser()
    )
    
    general_chain = (
        {"input": RunnablePassthrough()}
        | general_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Branch based on classification
    return RunnableBranch(
        (lambda x: "LEGAL_QUESTION" in x["category"], legal_question_chain),
        (lambda x: "STORY_ANALYSIS" in x["category"], story_analysis_chain),
        general_chain,
    )

# Main chain
def process_query(query: str) -> Dict[str, str]:
    """Process a user query through the Legal Bee RAG system."""
    # Detect language
    language = detect_language(query)
    
    # Classify input
    category = classifier_chain.invoke({"input": query}) # Pass as a dict
    
    # Create router chain
    router_chain = create_router_chain()
    
    # Process through appropriate chain
    response = router_chain.invoke({"input": query, "category": category})
    
    return {
        "query": query,
        "language": language,
        "category": category,
        "response": response
    }

# For testing
if __name__ == "__main__":
    # Test English query
    test_query = "What will the copyright law be called?"
    result = process_query(test_query)
    print(f"Query: {result['query']}")
    print(f"Language: {result['language']}")
    print(f"Category: {result['category']}")
    print(f"Response: {result['response']}")
    
    # Test Bangla query
    test_query_bn = "কপিরাইট এর  আইন কি নাম অভিহিত  হবে??"
    result_bn = process_query(test_query_bn)
    print(f"\nQuery: {result_bn['query']}")
    print(f"Language: {result_bn['language']}")
    print(f"Category: {result_bn['category']}")
    print(f"Response: {result_bn['response']}")