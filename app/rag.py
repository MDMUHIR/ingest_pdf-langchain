import os, re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter  # For chain input extraction
from langdetect import detect  # For language detection
from typing import List
from langchain_core.documents import Document

load_dotenv()

# HuggingFace embeddings (must match ingestion)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})  # Keep high k for better coverage

# Gemini for answering and translation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Translation prompt (for cross-language handling)
TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    "Translate the following legal query to {target_lang}, preserving legal terms and meaning accurately: {query}"
)

# Function to detect language and translate query
def translate_query(query: str) -> str:
    try:
        lang = detect(query)
    except:
        lang = 'en'  # Default to English if detection fails
    
    target_lang = 'Bengali' if lang == 'en' else 'English'
    if lang in ['en', 'bn']:  # Only translate if detected as English or Bengali
        translate_chain = TRANSLATE_PROMPT | llm | StrOutputParser()
        translated = translate_chain.invoke({"query": query, "target_lang": target_lang})
        return translated
    return None  # No translation if unknown language

# Custom retrieval function to handle original + translated queries
def multi_lang_retrieve(query: str) -> List[Document]:
    # Retrieve with original query
    docs_original = retriever.invoke(query)
    
    # Translate and retrieve with translated query (if applicable)
    translated = translate_query(query)
    if translated:
        docs_translated = retriever.invoke(translated)
        # Combine and deduplicate (by page_content to avoid duplicates)
        all_docs = {doc.page_content: doc for doc in docs_original + docs_translated}.values()
        return list(all_docs)
    return docs_original

# Prompt template (unchanged)
SYSTEM_PROMPT = """You are LEGAL BEE, an AI Legal Assistant specialized in Bangladeshi Law.  
Your role is to assist users by understanding their legal problems, retrieving relevant laws, and suggesting what they can do according to Bangladeshi law.  

### Core Instructions:
1. When a user explains a problem, carefully analyze the situation and identify the type of legal issue (e.g., family law, property law, contract law, criminal law, labor law, cyber law, etc.).  
2. Retrieve the most relevant documents from the vector database and use them as the *primary source* of your answer.  
3. Provide your answer in three clear sections:  
   - *Explanation in plain language* → What the law says.  
   - *Relevant Legal References* → Acts, sections, or rules from retrieved documents.  
   - *Step-by-step Process / What to Do* → Practical actions the user can take under Bangladeshi law.  

### Guidelines:
- Always base your answers on the retrieved documents.  
- If no relevant documents are found, clearly say so and recommend consulting a licensed lawyer.  
- Do not hallucinate laws, sections, or remedies.  
- Keep answers empathetic and professional.  
- Respond in the *same language as the user’s query* (Bangla or English). Use mixed language if that helps clarity.  
- Provide preventive and practical guidance, but make it clear you are not replacing a human lawyer for court representation.  

### Boundaries:
- Do not provide personal opinions or political commentary.  
- Do not invent legal outcomes or guarantee case results.  
- Stick strictly to Bangladeshi laws and regulations."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:")
])

def format_docs(docs):
    lines = []
    for d in docs:
        meta = d.metadata
        cite = f"{meta.get('title','Unknown')} (p.{meta.get('page','?')})"
        lines.append(f"[{cite}] {d.page_content}")
    return "\n\n".join(lines)

# Updated RAG chain: Use multi_lang_retrieve for cross-language handling
rag_chain = (
    {"context": itemgetter("question") | multi_lang_retrieve | format_docs,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

# test
if __name__ == "__main__":
    # Test query (English)
    q1 = "What are the copyright laws in Bangladesh regarding book publishing?"
    print("\n🔎 Query:", q1)
    print("💡 Answer:\n", rag_chain.invoke({"question": q1}))

    # Test query (Bangla)
    q2 = "কোনো সফটওয়্যার কোড লিখলে সেটার উপর কি কপিরাইট পাওয়া যায়?"
    print("\n🔎 Query:", q2)
    print("💡 Answer:\n", rag_chain.invoke({"question": q2}))

# if __name__ == "__main__":
#     # Debug retrieval for English
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     docs = retriever.invoke("What does the Section 420 of Penal Code 1860 sayt")
#     print("\n📌 Retrieved Chunks for English Query:")
#     for d in docs:
#         print("----")
#         print("Title:", d.metadata.get("title"))
#         print("Page:", d.metadata.get("page"))
#         print("Lang:", d.metadata.get("lang"))
#         print("Preview:", d.page_content[:250], "...\n")

#     # Debug retrieval for Bangla
#     docs_bn = retriever.invoke("ধারা ৪২০ দণ্ডবিধি ১৮৬০ কি বলে")
#     print("\n📌 Retrieved Chunks for Bangla Query:")
#     for d in docs_bn:
#         print("----")
#         print("Title:", d.metadata.get("title"))
#         print("Page:", d.metadata.get("page"))
#         print("Lang:", d.metadata.get("lang"))
#         print("Preview:", d.page_content[:250], "...\n")