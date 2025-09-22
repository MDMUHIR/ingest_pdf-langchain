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
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever setup
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb, namespace="_default_")

# OpenRouter LLM
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=SecretStr(OPENROUTER_API_KEY),
    base_url="https://openrouter.ai/api/v1"
)

# Enhanced language detection with fallback
def detect_language(query: str) -> str:
    """Detect language with better handling of mixed content"""
    try:
        detected = detect(query)
        # Additional checks for Bengali characters
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        
        if bengali_chars > english_chars:
            return 'bn'
        elif english_chars > bengali_chars:
            return 'en'
        else:
            return detected
    except LangDetectException:
        # Fallback to character-based detection
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        return 'bn' if bengali_chars > english_chars else 'en'

# Enhanced translation prompts
TRANSLATE_TO_ENGLISH_PROMPT = ChatPromptTemplate.from_template(
    """Translate this Bengali legal query to English. Keep legal terms, act names, and section numbers as they are or use standard English equivalents. Maintain the legal context and meaning precisely.

Bengali Query: {query}

English Translation:"""
)

TRANSLATE_TO_BENGALI_PROMPT = ChatPromptTemplate.from_template(
    """Translate this English legal query to Bengali. Keep legal terms, act names, and section numbers as they are or use standard Bengali equivalents. Maintain the legal context and meaning precisely.

English Query: {query}

Bengali Translation:"""
)

# Enhanced query translation
def translate_query(query: str, source_lang: str, target_lang: str) -> str:
    """Translate query between Bengali and English"""
    if source_lang == target_lang:
        return query
    
    try:
        if target_lang == 'en' and source_lang == 'bn':
            chain = TRANSLATE_TO_ENGLISH_PROMPT | llm | StrOutputParser()
            return chain.invoke({"query": query}).strip()
        elif target_lang == 'bn' and source_lang == 'en':
            chain = TRANSLATE_TO_BENGALI_PROMPT | llm | StrOutputParser()
            return chain.invoke({"query": query}).strip()
        else:
            return query
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return query

# Enhanced multi-language retrieval with better scoring
def enhanced_multi_lang_retrieve(query: str, k: int = 30) -> Tuple[List[Document], str]:
    """Enhanced retrieval that searches in both languages and deduplicates properly"""
    detected_lang = detect_language(query)
    logger.info(f"Detected language: {detected_lang}")
    
    all_docs = []
    
    # Search with original query
    try:
        original_docs = vectorstore.similarity_search(query, k=k)
        for doc in original_docs:
            doc.metadata['search_lang'] = detected_lang
            doc.metadata['search_type'] = 'original'
        all_docs.extend(original_docs)
        logger.info(f"Retrieved {len(original_docs)} docs with original query")
    except Exception as e:
        logger.error(f"Original query search failed: {e}")
    
    # Search with translated query
    target_lang = 'en' if detected_lang == 'bn' else 'bn'
    translated_query = translate_query(query, detected_lang, target_lang)
    
    if translated_query != query:
        try:
            translated_docs = vectorstore.similarity_search(translated_query, k=k)
            for doc in translated_docs:
                doc.metadata['search_lang'] = target_lang
                doc.metadata['search_type'] = 'translated'
            all_docs.extend(translated_docs)
            logger.info(f"Retrieved {len(translated_docs)} docs with translated query: {translated_query}")
        except Exception as e:
            logger.error(f"Translated query search failed: {e}")
    
    # Enhanced deduplication based on content similarity and metadata
    unique_docs = {}
    for doc in all_docs:
        # Create a composite key for deduplication
        doc_id = doc.metadata.get('doc_id')
        page = doc.metadata.get('page', 0)
        section = doc.metadata.get('section', '')
        
        # Use doc_id if available, otherwise use content hash
        if doc_id:
            key = f"{doc_id}_{page}_{section}"
        else:
            key = f"{hash(doc.page_content)}_{page}_{section}"
        
        if key not in unique_docs:
            unique_docs[key] = doc
        else:
            # Keep the one from the original language search if possible
            existing = unique_docs[key]
            if (doc.metadata.get('search_type') == 'original' and 
                existing.metadata.get('search_type') == 'translated'):
                unique_docs[key] = doc
    
    final_docs = list(unique_docs.values())
    logger.info(f"Final unique documents: {len(final_docs)}")
    
    return final_docs, detected_lang

# Enhanced document formatting with better language handling
def format_docs_enhanced(docs: List[Document], query_lang: str) -> str:
    """Format documents with proper language handling"""
    if not docs:
        return "No relevant documents found."
    
    lines = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        
        # Get title in appropriate language
        if query_lang == 'bn':
            title = meta.get('title-bn', meta.get('title', 'Unknown'))
            if not title or title == 'Unknown':
                title = meta.get('title-en', 'Unknown')
        else:
            title = meta.get('title-en', meta.get('title', 'Unknown'))
            if not title or title == 'Unknown':
                title = meta.get('title-bn', 'Unknown')
        
        # Format other metadata
        act_num = meta.get('act-number', '')
        section = meta.get('section', '')
        page = meta.get('page', '?')
        
        # Create citation
        citation_parts = [title]
        if act_num:
            citation_parts.append(act_num)
        if section:
            citation_parts.append(f"Section {section}")
        citation_parts.append(f"p.{page}")
        
        cite = " - ".join(citation_parts)
        
        # Format content
        content = doc.page_content.strip()
        if len(content) > 500:  # Truncate very long content
            content = content[:500] + "..."
        
        lines.append(f"[{cite}]\n{content}")
    
    return "\n\n".join(lines)

# Language-specific system prompts
SYSTEM_PROMPT_EN = """You are LEGAL BEE, an AI Legal Assistant specialized in Bangladeshi Law.

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
- Respond in English since the user asked in English.
- Provide preventive guidance but clarify you are not a substitute for a human lawyer.

### Boundaries:
- No personal opinions or political commentary.
- No invented outcomes or case guarantees.
- Stick to Bangladeshi laws only."""

SYSTEM_PROMPT_BN = """আপনি LEGAL BEE, একজন AI আইনি সহায়ক যিনি বাংলাদেশী আইনে বিশেষজ্ঞ।

### মূল নির্দেশাবলী:
1. ব্যবহারকারীর প্রশ্ন বিশ্লেষণ করে আইনি সমস্যা চিহ্নিত করুন (যেমন, পারিবারিক আইন, সম্পত্তি আইন, চুক্তি আইন, ফৌজদারি আইন, শ্রম আইন, সাইবার আইন ইত্যাদি)।
2. প্রাপ্ত নথিগুলিকে প্রাথমিক উৎস হিসেবে ব্যবহার করুন। [শিরোনাম - আইন নম্বর - ধারা (পৃ.পেজ)] ফরম্যাটে সঠিকভাবে উদ্ধৃত করুন।
3. আপনার উত্তর তিনটি অংশে সাজান:
   - **সহজ ভাষায় ব্যাখ্যা**: আইন কী বলে তা সরল ভাষায় সংক্ষেপে বলুন।
   - **প্রাসঙ্গিক আইনি রেফারেন্স**: প্রাপ্ত নথি থেকে আইন, ধারা, বা নিয়ম উদ্ধৃতি সহ তালিকা করুন।
   - **ধাপে ধাপে প্রক্রিয়া / কী করণীয়**: বাংলাদেশী আইনের অধীনে ব্যবহারিক পদক্ষেপ প্রদান করুন।

### নির্দেশিকা:
- উত্তর কঠোরভাবে প্রাপ্ত নথির উপর ভিত্তি করুন। কোনটি প্রাসঙ্গিক না হলে তা উল্লেখ করুন এবং লাইসেন্সপ্রাপ্ত আইনজীবীর পরামর্শ নিতে বলুন।
- আইন, ধারা, বা প্রতিকার বানিয়ে বলবেন না।
- সহানুভূতিশীল, পেশাদার এবং নিরপেক্ষ থাকুন।
- ব্যবহারকারী বাংলায় প্রশ্ন করেছেন তাই বাংলায় উত্তর দিন।
- প্রতিরোধমূলক নির্দেশনা দিন কিন্তু স্পষ্ট করুন যে আপনি মানব আইনজীবীর বিকল্প নন।

### সীমানা:
- কোনো ব্যক্তিগত মতামত বা রাজনৈতিক মন্তব্য নয়।
- কোনো কল্পিত ফলাফল বা মামলার গ্যারান্টি নয়।
- শুধুমাত্র বাংলাদেশী আইনে সীমাবদ্ধ থাকুন।"""

# Enhanced input preparation
def prepare_input_enhanced(question: str) -> Dict[str, str]:
    """Enhanced input preparation with better language handling"""
    docs, detected_lang = enhanced_multi_lang_retrieve(question)
    context = format_docs_enhanced(docs, detected_lang)
    
    return {
        "context": context,
        "question": question,
        "language": detected_lang
    }

# Language-aware prompt selection
def get_prompt_for_language(lang: str) -> ChatPromptTemplate:
    """Get appropriate prompt based on detected language"""
    if lang == 'bn':
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_BN),
            ("user", "প্রশ্ন: {question}\n\nপ্রসঙ্গ:\n{context}\n\nউদ্ধৃতি সহ উত্তর দিন:")
        ])
    else:
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_EN),
            ("user", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:")
        ])

# Enhanced query classifier
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """Classify this legal query as either:
    - 'story': Personal legal problem or scenario requiring fact extraction
    - 'direct': Direct question about laws/legal procedures

    Query: {query}
    
    Consider the language and context. Output only: story or direct"""
)

classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()

# Fact extraction prompt
FACT_EXTRACT_PROMPT_EN = ChatPromptTemplate.from_template(
    """Extract the key legal facts and issues from this personal legal story. Focus on actionable legal elements.

    Story: {query}
    
    Key Legal Facts:"""
)

FACT_EXTRACT_PROMPT_BN = ChatPromptTemplate.from_template(
    """এই ব্যক্তিগত আইনি গল্প থেকে মূল আইনি তথ্য এবং সমস্যাগুলি বের করুন। কার্যকর আইনি উপাদানগুলিতে মনোনিবেশ করুন।

    গল্প: {query}
    
    মূল আইনি তথ্য:"""
)

# Enhanced invocation function
def enhanced_invoke(query: str) -> str:
    """Enhanced invocation with better language handling and retrieval"""
    try:
        # Detect language first
        detected_lang = detect_language(query)
        logger.info(f"Processing query in {detected_lang}")
        
        # Classify query type
        query_type = classifier_chain.invoke({"query": query}).strip().lower()
        logger.info(f"Query type: {query_type}")
        
        # Handle story-type queries with fact extraction
        if query_type == 'story':
            if detected_lang == 'bn':
                fact_extract_chain = FACT_EXTRACT_PROMPT_BN | llm | StrOutputParser()
            else:
                fact_extract_chain = FACT_EXTRACT_PROMPT_EN | llm | StrOutputParser()
            
            facts = fact_extract_chain.invoke({"query": query})
            
            # Create enhanced query combining facts and original
            if detected_lang == 'bn':
                adjusted_query = f"তথ্যের ভিত্তিতে আইনি সমস্যা: {facts}\nমূল প্রশ্ন: {query}"
            else:
                adjusted_query = f"Legal issue based on facts: {facts}\nOriginal query: {query}"
        else:
            adjusted_query = query
        
        # Prepare input with enhanced retrieval
        input_data = prepare_input_enhanced(adjusted_query)
        
        # Get appropriate prompt for the detected language
        prompt = get_prompt_for_language(detected_lang)
        
        # Create and run the chain
        rag_chain = (
            RunnableLambda(lambda x: x)
            | prompt
            | llm
            | StrOutputParser()
        )
        
        result = rag_chain.invoke(input_data)
        
        # Add fallback message if no relevant docs found
        if "No relevant documents found" in input_data["context"]:
            fallback_msg = (
                "\n\n⚠️ দুঃখিত, এই প্রশ্নের জন্য আমার ডেটাবেসে প্রাসঙ্গিক তথ্য পাওয়া যায়নি। একজন লাইসেন্সপ্রাপ্ত আইনজীবীর সাথে পরামর্শ করার পরামর্শ দিচ্ছি।" 
                if detected_lang == 'bn' else 
                "\n\n⚠️ Sorry, no relevant information was found in my database for this query. I recommend consulting with a licensed lawyer."
            )
            result += fallback_msg
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced_invoke: {e}")
        error_msg = (
            f"দুঃখিত, আপনার প্রশ্ন প্রক্রিয়া করতে একটি সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন। ত্রুটি: {str(e)}"
            if detect_language(query) == 'bn' else
            f"Sorry, there was an error processing your query. Please try again. Error: {str(e)}"
        )
        return error_msg

# Test function
def test_rag():
    """Test the enhanced RAG with both languages"""
    # Test queries
    test_queries = [
        "Under the Election Officers (Special Provisions) Act, 1991 of Bangladesh, what legal actions can be taken if an election officer is found to be biased?",
        "বাংলাদেশের নির্বাচন কর্মকর্তা (বিশেষ বিধান) আইন, ১৯৯১ অনুসারে, যদি কোনও নির্বাচন কর্মকর্তা পক্ষপাতদুষ্ট হন তবে কী আইনি ব্যবস্থা নেওয়া যেতে পারে?",
        "আমার ব্যাংক আমার বিরুদ্ধে মিথ্যা মামলা করেছে। আমি কী করতে পারি?",
        "My bank has filed a false case against me. What can I do?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*50}")
        result = enhanced_invoke(query)
        print(result)

if __name__ == "__main__":
    test_rag()