import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from langdetect import detect, LangDetectException
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# HuggingFace embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever setup
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb, namespace="_default_")
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

# Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# OpenRouter LLM
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Corrected model ID for OpenRouter
    temperature=0,
    api_key=SecretStr(OPENROUTER_API_KEY),
    base_url="https://openrouter.ai/api/v1"
)

# Translation prompt
TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    """Translate the following legal query to {target_lang}. 
    Preserve legal terms (e.g., section numbers, act names) in their original form or standard transliteration.
    Maintain accurate meaning, structure, and formality: {query}"""
)

# Language detection
def detect_language(query: str) -> str:
    try:
        return detect(query)
    except LangDetectException:
        return 'en'

# Query translation
def translate_query(query: str, detected_lang: str) -> str:
    if detected_lang in ['en', 'bn']:
        target_lang = 'bn' if detected_lang == 'en' else 'en'
        translate_chain = TRANSLATE_PROMPT | llm | StrOutputParser()
        translated = translate_chain.invoke({"query": query, "target_lang": target_lang})
        return translated
    return None

# Multi-language retrieval
def multi_lang_retrieve(query: str) -> List[Document]:
    detected_lang = detect_language(query)
    docs_original = retriever.invoke(query)
    translated = translate_query(query, detected_lang)
    docs_translated = retriever.invoke(translated) if translated else []
    unique_docs: Dict[str, Document] = {}
    for doc in docs_original + docs_translated:
        doc_id = doc.metadata.get('doc_id', hash(doc.page_content))
        if doc_id not in unique_docs:
            unique_docs[str(doc_id)] = doc
    return list(unique_docs.values())

# System prompt
SYSTEM_PROMPT = """You are LEGAL BEE, an AI Legal Assistant specialized in Bangladeshi Law.  
Your role is to assist users by understanding their legal problems, retrieving relevant laws, and suggesting actions according to Bangladeshi law.  

### Core Instructions:
1. Analyze the user's query to identify the legal issue (e.g., family law, property law, contract law, criminal law, labor law, cyber law, etc.).  
2. Use retrieved documents as the primary source. Cite them accurately using the format [Title - Act Number - Section (p.Page)].
3. Structure your answer in three sections:  
   - **Explanation in Plain Language**: Summarize what the law says in simple terms.  
   - **Relevant Legal References**: List acts, sections, or rules from retrieved documents with citations.  
   - **Step-by-Step Process / What to Do**: Provide practical actions under Bangladeshi law.  

### Guidelines:
- Base answers strictly on retrieved documents. If none are relevant, state so and recommend consulting a licensed lawyer.  
- Do not hallucinate laws, sections, or remedies.  
- Be empathetic, professional, and neutral.  
- Respond in the same language as the user’s query (Bengali or English). Use mixed language only if it enhances clarity.  
- Provide preventive guidance but clarify you are not a substitute for a human lawyer.  
- If the query is a personal story, extract key facts and map to legal categories before responding.

### Boundaries:
- No personal opinions or political commentary.  
- No invented outcomes or case guarantees.  
- Stick to Bangladeshi laws only."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:")
])

# Format documents for context
def format_docs(docs: List[Document], query_lang: str) -> str:
    lines = []
    for d in docs:
        meta = d.metadata
        title = meta.get('title-bn', meta.get('title', 'Unknown')) if query_lang == 'bn' else meta.get('title-en', meta.get('title', 'Unknown'))
        act_num = meta.get('act-number', '')
        section = meta.get('section', '')
        page = meta.get('page', '?')
        cite = f"{title} - {act_num} - {section} (p.{page})"
        lines.append(f"[{cite}] {d.page_content}")
    return "\n\n".join(lines)

# Input preparation function
def prepare_input(question: str) -> Dict[str, str]:
    detected_lang = detect_language(question)
    docs = multi_lang_retrieve(question)
    context = format_docs(docs, detected_lang)
    return {"context": context, "question": question}

# Query classifier
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """Classify the query as:
    - 'story': If it's a personal legal problem or scenario.
    - 'direct': If it's a straightforward question about laws.
    Query: {query}
    Output only: story or direct"""
)

classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()

# Enhanced invocation
def enhanced_invoke(query: str) -> str:
    query_type = classifier_chain.invoke({"query": query}).strip().lower()
    if query_type == 'story':
        fact_extract_prompt = ChatPromptTemplate.from_template(
            "Extract key legal facts from this story: {query}\nFacts:"
        )
        facts = (fact_extract_prompt | llm | StrOutputParser()).invoke({"query": query})
        adjusted_query = f"Legal issue based on facts: {facts}\nOriginal query: {query}"
    else:
        adjusted_query = query
    
    # Use RunnableLambda to prepare input
    rag_chain = (
        RunnableLambda(lambda x: prepare_input(x["question"] if isinstance(x, dict) else getattr(x, "question", "")))
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke({"question": adjusted_query})

# Test
if __name__ == "__main__":
    # Test query (English)
    q1 = "Under the Election Officers (Special Provisions) Act, 1991 of Bangladesh, what legal actions can be taken if an election officer is found to be biased or fails to perform their duties impartially during an election?"
    print("\n🔎 Query:", q1)
    print("💡 Answer:\n", enhanced_invoke(q1))

    # Test query (Bangla)
    q2 = "বাংলাদেশের নির্বাচন কর্মকর্তা (বিশেষ বিধান) আইন, ১৯৯১ অনুসারে, যদি কোনও নির্বাচন কর্মকর্তা নির্বাচনের সময় পক্ষপাতদুষ্ট হন বা নিরপেক্ষভাবে তাদের দায়িত্ব পালনে ব্যর্থ হন তবে কী আইনি ব্যবস্থা নেওয়া যেতে পারে?"
    print("\n🔎 Query:", q2)
    print("💡 Answer:\n", enhanced_invoke(q2))
    
# if __name__ == "__main__":
#     # Test query (English)
#     q1 = "Under the Bank Companies Act, 1991 of Bangladesh, if a bank intentionally fails to take legal action against willful defaulters, what legal remedies are available to the customer or other concerned parties?"
#     print("\n🔎 Query:", q1)
#     print("💡 Answer:\n", enhanced_invoke(q1))

#     # Test query (Bangla)
#     q2 = "বাংলাদেশের ব্যাংক-কোম্পানী আইন, ১৯৯১ অনুযায়ী যদি কোনো ব্যাংক ইচ্ছাকৃতভাবে ঋণ খেলাপিদের বিরুদ্ধে আইনগত ব্যবস্থা না নেয়, তাহলে সেই ক্ষেত্রে গ্রাহক বা সংশ্লিষ্ট পক্ষের কী ধরনের আইনগত প্রতিকার পাওয়ার সুযোগ আছে?"
#     print("\n🔎 Query:", q2)
#     print("💡 Answer:\n", enhanced_invoke(q2))