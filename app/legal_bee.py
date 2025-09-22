import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter
from langdetect import detect, LangDetectException
from typing import List, Dict, Tuple, Set, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import logging
import numpy as np

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever setup
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb, namespace="")

# OpenRouter LLM
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=SecretStr(OPENROUTER_API_KEY),
    base_url="https://openrouter.ai/api/v1"
)

# Enhanced language detection with confidence scoring
def detect_language_with_confidence(query: str) -> Tuple[str, float]:
    """Enhanced language detection with confidence scoring"""
    try:
        detected = detect(query)
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        total_chars = bengali_chars + english_chars
        
        if total_chars == 0:
            return 'en', 0.5
        
        bengali_ratio = bengali_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if bengali_ratio > 0.6:
            return 'bn', bengali_ratio
        elif english_ratio > 0.6:
            return 'en', english_ratio
        else:
            return detected, 0.7
    except LangDetectException:
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        if bengali_chars > english_chars:
            return 'bn', 0.6
        else:
            return 'en', 0.6

# Legal domain keyword extraction
LEGAL_KEYWORDS_EN = {
    'criminal': ['murder', 'theft', 'fraud', 'assault', 'criminal', 'police', 'fir', 'charge', 'arrest', 'bail', 'trial'],
    'civil': ['contract', 'property', 'land', 'dispute', 'damages', 'civil', 'suit', 'decree', 'injunction'],
    'family': ['marriage', 'divorce', 'custody', 'alimony', 'family', 'wife', 'husband', 'child', 'adoption'],
    'banking': ['bank', 'loan', 'defaulter', 'account', 'deposit', 'credit', 'banking', 'foreclosure', 'recovery'],
    'election': ['election', 'vote', 'candidate', 'officer', 'commission', 'ballot', 'petition'],
    'labor': ['worker', 'employee', 'salary', 'wage', 'labor', 'employment', 'job', 'termination', 'compensation'],
    'cyber': ['internet', 'online', 'digital', 'cyber', 'computer', 'data', 'hacking', 'privacy', 'defamation'],
    'constitutional': ['constitution', 'rights', 'fundamental', 'writ', 'petition', 'supreme court'],
    'administrative': ['administrative', 'tribunal', 'service', 'government', 'order', 'appeal'],
    'environmental': ['environment', 'pollution', 'forest', 'wildlife', 'conservation']
}

LEGAL_KEYWORDS_BN = {
    'criminal': ['খুন', 'চুরি', 'প্রতারণা', 'আক্রমণ', 'ফৌজদারি', 'পুলিশ', 'মামলা', 'গ্রেফতার', 'জামিন', 'বিচার'],
    'civil': ['চুক্তি', 'সম্পত্তি', 'জমি', 'বিরোধ', 'ক্ষতিপূরণ', 'দেওয়ানি', 'মামলা', 'ডিক্রি', 'নিষেধাজ্ঞা'],
    'family': ['বিবাহ', 'তালাক', 'অভিভাবকত্ব', 'ভরণপোষণ', 'পারিবারিক', 'স্ত্রী', 'স্বামী', 'সন্তান', 'দত্তক'],
    'banking': ['ব্যাংক', 'ঋণ', 'খেলাপি', 'অ্যাকাউন্ট', 'আমানত', 'ঋণ', 'ব্যাংকিং', 'ফোরক্লোজার', 'আদায়'],
    'election': ['নির্বাচন', 'ভোট', 'প্রার্থী', 'কর্মকর্তা', 'কমিশন', 'ব্যালট', 'পিটিশন'],
    'labor': ['শ্রমিক', 'কর্মচারী', 'বেতন', 'মজুরি', 'শ্রম', 'চাকরি', 'কাজ', 'বরখাস্ত', 'ক্ষতিপূরণ'],
    'cyber': ['ইন্টারনেট', 'অনলাইন', 'ডিজিটাল', 'সাইবার', 'কম্পিউটার', 'তথ্য', 'হ্যাকিং', 'গোপনীয়তা', 'মানহানি'],
    'constitutional': ['সংবিধান', 'অধিকার', 'মৌলিক', 'রিট', 'পিটিশন', 'সুপ্রিম কোর্ট'],
    'administrative': ['প্রশাসনিক', 'ট্রাইব্যুনাল', 'সার্ভিস', 'সরকার', 'আদেশ', 'আপিল'],
    'environmental': ['পরিবেশ', 'দূষণ', 'বন', 'বন্যপ্রাণী', 'সংরক্ষণ']
}

def extract_legal_domain(query: str, lang: str) -> List[str]:
    """Extract legal domains from query"""
    keywords = LEGAL_KEYWORDS_BN if lang == 'bn' else LEGAL_KEYWORDS_EN
    query_lower = query.lower()
    
    domains = []
    for domain, words in keywords.items():
        if any(word in query_lower for word in words):
            domains.append(domain)
    
    return domains if domains else ['general']

# Advanced query expansion
def expand_query_terms(query: str, lang: str) -> List[str]:
    """Expand query with related legal terms"""
    expansions = []
    domains = extract_legal_domain(query, lang)
    
    for domain in domains:
        if domain in LEGAL_KEYWORDS_EN:
            if lang == 'bn':
                expansions.extend(LEGAL_KEYWORDS_BN.get(domain, [])[:3])
            else:
                expansions.extend(LEGAL_KEYWORDS_EN.get(domain, [])[:3])
    
    return list(set(expansions))

# Enhanced translation with legal context
ENHANCED_TRANSLATE_TO_ENGLISH_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal translation expert. Translate this Bengali legal query to English while preserving all legal nuances and terminology.

Bengali Query: {query}
Legal Domain: {domain}

Precise English Translation:"""
)

ENHANCED_TRANSLATE_TO_BENGALI_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal translation expert. Translate this English legal query to Bengali while preserving all legal nuances and terminology.

English Query: {query}
Legal Domain: {domain}

Precise Bengali Translation:"""
)

def enhanced_translate_query(query: str, source_lang: str, target_lang: str) -> str:
    """Enhanced translation with legal domain awareness"""
    if source_lang == target_lang:
        return query
    
    try:
        domains = extract_legal_domain(query, source_lang)
        domain_context = ', '.join(domains)
        
        if target_lang == 'en' and source_lang == 'bn':
            chain = ENHANCED_TRANSLATE_TO_ENGLISH_PROMPT | llm | StrOutputParser()
            return chain.invoke({"query": query, "domain": domain_context}).strip()
        elif target_lang == 'bn' and source_lang == 'en':
            chain = ENHANCED_TRANSLATE_TO_BENGALI_PROMPT | llm | StrOutputParser()
            return chain.invoke({"query": query, "domain": domain_context}).strip()
        else:
            return query
    except Exception as e:
        logger.warning(f"Enhanced translation failed: {e}")
        return query

# Multi-strategy retrieval system
class MultiStrategyRetriever:
    def __init__(self, vectorstore, k=15, min_score=0.65):
        self.vectorstore = vectorstore
        self.k = k
        self.min_score = min_score
    
    def hybrid_search(self, query: str, lang: str) -> List[Document]:
        """Hybrid search with lang filter"""
        all_results = []
        
        # Lang filter
        lang_filter = {"lang": "bn / en"} if lang == 'bn' else {"lang": "en / bn"}
        
        def perform_search(search_query: str, strategy: str, score_multiplier: float = 1.0):
            try:
                search_kwargs = {'k': self.k, 'filter': lang_filter, 'score_threshold': self.min_score}
                results = self.vectorstore.similarity_search_with_score(search_query, **search_kwargs)
                filtered = [(doc, score * score_multiplier) for doc, score in results]
                for doc, score in filtered:
                    doc.metadata['retrieval_strategy'] = strategy
                    doc.metadata['relevance_score'] = score
                return [doc for doc, _ in filtered]
            except Exception as e:
                logger.error(f"Search for '{search_query}' failed: {e}")
                return []
        
        # Direct search
        direct_docs = perform_search(query, 'direct', 1.0)
        all_results.extend(direct_docs)
        
        # Domain-specific search
        domains = extract_legal_domain(query, lang)
        for domain in domains:
            domain_keywords = LEGAL_KEYWORDS_BN.get(domain, []) if lang == 'bn' else LEGAL_KEYWORDS_EN.get(domain, [])
            for keyword in domain_keywords[:2]:
                keyword_docs = perform_search(keyword, f'domain_{domain}', 0.85)
                all_results.extend(keyword_docs)
        
        # Translated query search
        target_lang = 'en' if lang == 'bn' else 'bn'
        translated_query = enhanced_translate_query(query, lang, target_lang)
        if translated_query != query:
            translated_docs = perform_search(translated_query, 'translated', 0.95)
            all_results.extend(translated_docs)
        
        # Expanded terms
        expanded_terms = expand_query_terms(query, lang)
        for term in expanded_terms[:3]:
            expanded_docs = perform_search(term, 'expanded', 0.75)
            all_results.extend(expanded_docs)
        
        return all_results
    
    def intelligent_deduplication(self, docs: List[Document]) -> List[Document]:
        """Intelligent deduplication with relevance scoring"""
        unique_docs = {}
        
        for doc in docs:
            key = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}_{doc.metadata.get('section', '')}"
            
            if key not in unique_docs or doc.metadata.get('relevance_score', 0) > unique_docs[key].metadata.get('relevance_score', 0):
                unique_docs[key] = doc
        
        return sorted(unique_docs.values(), key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)[:10]

# Initialize enhanced retriever
multi_retriever = MultiStrategyRetriever(vectorstore)

# Enhanced document relevance scoring
def score_document_relevance(doc: Document, query: str, lang: str) -> float:
    """Score document relevance based on multiple factors"""
    score = doc.metadata.get('relevance_score', 0.5) * 0.5
    content = doc.page_content.lower()
    query_lower = query.lower()
    
    query_words = set(query_lower.split())
    content_words = set(content.split())
    common_words = query_words.intersection(content_words)
    if query_words:
        score += (len(common_words) / len(query_words)) * 0.2
    
    domains = extract_legal_domain(query, lang)
    for domain in domains:
        keywords = LEGAL_KEYWORDS_BN.get(domain, []) if lang == 'bn' else LEGAL_KEYWORDS_EN.get(domain, [])
        domain_matches = sum(1 for keyword in keywords if keyword in content)
        if keywords:
            score += (domain_matches / len(keywords)) * 0.2
    
    return min(score, 1.0)

# Enhanced context formatting
def format_context_intelligently(docs: List[Document], query: str, lang: str) -> str:
    """Format context with intelligent ranking"""
    if not docs:
        return "No relevant documents found."
    
    scored_docs = [(doc, score_document_relevance(doc, query, lang)) for doc in docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    formatted_sections = []
    for doc, score in scored_docs[:8]:
        meta = doc.metadata
        title = meta.get('title-bn', meta.get('title', 'Unknown')) if lang == 'bn' else meta.get('title-en', meta.get('title', 'Unknown'))
        
        act_key = meta.get('act-number', '')
        section = meta.get('section', '')
        page = meta.get('page', '?')
        cite = f"{title} - {act_key} - Section {section} (p.{page})" if section else f"{title} - {act_key} (p.{page})"
        
        content = doc.page_content.strip()
        if len(content) > 300:
            content = content[:300] + "..."
        
        formatted_sections.append(f"[{cite}]\n{content}")
    
    return "\n\n".join(formatted_sections)

# Enhanced system prompts with professional solution focus
ENHANCED_SYSTEM_PROMPT_EN = """You are LEGAL BEE, an expert AI Legal Assistant specializing in Bangladeshi law. Provide concise, professional, and exact solutions based on retrieved documents.

### Response Structure (MANDATORY):
**🎯 LEGAL ISSUE IDENTIFICATION**
- Identify the core legal problem
- Specify the legal domain
- Note urgency if applicable

**📚 APPLICABLE LAW & ANALYSIS**
- Cite exact laws/sections [Title - Act Number - Section (p.Page)]
- Briefly explain relevance
- Analyze application to the situation

**⚖️ LEGAL SOLUTION**
- Provide the exact legal remedy/solution
- Detail specific rights and actions
- Outline potential outcomes

**📋 ACTION PLAN**
- List immediate steps
- Required documents
- Legal process overview
- Key deadlines

**⚠️ CONSIDERATIONS**
- Highlight key risks
- Note time limits
- Suggest legal consultation if needed

### Guidelines:
- Focus on providing a clear, actionable solution
- Use precise citations from documents
- Keep responses concise and professional
- Avoid general advice; base on specific law
- Recommend lawyer consultation for complex cases"""

ENHANCED_SYSTEM_PROMPT_BN = """আপনি LEGAL BEE, বাংলাদেশী আইনে বিশেষজ্ঞ AI আইনি সহায়ক। উদ্ধৃত নথিগুলোর ভিত্তিতে সংক্ষিপ্ত, পেশাদার এবং সঠিক সমাধান প্রদান করুন।

### উত্তরের কাঠামো (বাধ্যতামূলক):
**🎯 আইনি সমস্যা চিহ্নিতকরণ**
- মূল আইনি সমস্যা চিহ্নিত করুন
- আইনের ক্ষেত্র নির্দিষ্ট করুন
- জরুরি হলে উল্লেখ করুন

**📚 প্রযোজ্য আইন ও বিশ্লেষণ**
- নির্দিষ্ট আইন/ধারা উদ্ধৃত করুন [শিরোনাম - আইন নম্বর - ধারা (পৃ.পেজ)]
- সংক্ষেপে প্রাসঙ্গিকতা ব্যাখ্যা করুন
- পরিস্থিতিতে প্রয়োগ বিশ্লেষণ করুন

**⚖️ আইনি সমাধান**
- সঠিক আইনি প্রতিকার/সমাধান প্রদান করুন
- নির্দিষ্ট অধিকার ও পদক্ষেপ বিস্তারিত করুন
- সম্ভাব্য ফলাফল বর্ণনা করুন

**📋 কর্মপরিকল্পনা**
- তাৎক্ষণিক পদক্ষেপ তালিকাভুক্ত করুন
- প্রয়োজনীয় কাগজপত্র
- আইনি প্রক্রিয়ার সংক্ষিপ্ত বিবরণ
- গুরুত্বপূর্ণ সময়সীমা

**⚠️ বিবেচনা**
- মূল ঝুঁকি তুলে ধরুন
- সময়সীমা নোট করুন
- জটিল ক্ষেত্রে আইনজীবীর পরামর্শ সুপারিশ করুন

### নির্দেশনা:
- স্পষ্ট, কার্যকর সমাধানে ফোকাস করুন
- নথিগুলো থেকে নির্দিষ্ট উদ্ধৃতি ব্যবহার করুন
- উত্তর সংক্ষিপ্ত ও পেশাদার রাখুন
- সাধারণ পরামর্শ এড়িয়ে নির্দিষ্ট আইনে ভিত্তি করুন
- জটিল ক্ষেত্রে আইনজীবীর পরামর্শ সুপারিশ করুন"""

# Enhanced fact extraction
ENHANCED_FACT_EXTRACT_PROMPT_EN = ChatPromptTemplate.from_template(
    """As a legal expert, extract key facts from this situation for case analysis. Be concise.

Situation: {query}

Provide brief:
1. KEY LEGAL FACTS
2. PARTIES INVOLVED
3. LEGAL ISSUES
4. RELEVANT DATES
5. EVIDENCE
6. DESIRED OUTCOME

Extracted Facts:"""
)

ENHANCED_FACT_EXTRACT_PROMPT_BN = ChatPromptTemplate.from_template(
    """একজন আইনি বিশেষজ্ঞ হিসেবে, এই পরিস্থিতি থেকে মূল তথ্য বের করুন মামলা বিশ্লেষণের জন্য। সংক্ষিপ্ত হোন।

পরিস্থিতি: {query}

সংক্ষিপ্তভাবে প্রদান করুন:
১. মূল আইনি তথ্য
২. জড়িত পক্ষসমূহ
৩. আইনি সমস্যা
৪. প্রাসঙ্গিক তারিখ
৫. প্রমাণ
৬. কাঙ্ক্ষিত ফলাফল

নিষ্কাশিত তথ্য:"""
)

# Advanced query processing
def process_query_with_legal_intelligence(query: str) -> Dict[str, Any]:
    """Process query with legal domain intelligence"""
    lang, confidence = detect_language_with_confidence(query)
    domains = extract_legal_domain(query, lang)
    is_urgent = any(indicator in query.lower() for indicator in ['urgent', 'জরুরি', 'immediate', 'তাৎক্ষণিক'])
    return {'language': lang, 'domains': domains, 'is_urgent': is_urgent}

# Enhanced legal processing
def enhanced_legal_processing(query: str) -> str:
    """Process legal query or situation with professional solution"""
    try:
        # Query analysis
        query_analysis = process_query_with_legal_intelligence(query)
        lang = query_analysis['language']
        
        # Classify query type
        classifier_prompt = ChatPromptTemplate.from_template(
            """Classify this as 'situation' (personal case) or 'question' (legal query).
            Query: {query}
            Classification (one word):"""
        )
        classifier_chain = classifier_prompt | llm | StrOutputParser()
        query_type = classifier_chain.invoke({"query": query}).strip().lower()
        
        # Process situation or question
        processed_query = query
        if query_type == 'situation':
            if lang == 'bn':
                fact_chain = ENHANCED_FACT_EXTRACT_PROMPT_BN | llm | StrOutputParser()
            else:
                fact_chain = ENHANCED_FACT_EXTRACT_PROMPT_EN | llm | StrOutputParser()
            extracted_facts = fact_chain.invoke({"query": query})
            if lang == 'bn':
                processed_query = f"পরিস্থিতি: {extracted_facts}\nপ্রশ্ন: {query}"
            else:
                processed_query = f"Situation: {extracted_facts}\nQuestion: {query}"
        
        # Document retrieval
        retrieved_docs = multi_retriever.hybrid_search(processed_query, lang)
        final_docs = multi_retriever.intelligent_deduplication(retrieved_docs)
        formatted_context = format_context_intelligently(final_docs, processed_query, lang)
        
        # Generate response
        if lang == 'bn':
            prompt = ChatPromptTemplate.from_messages([
                ("system", ENHANCED_SYSTEM_PROMPT_BN),
                ("user", "প্রশ্ন/পরিস্থিতি: {question}\n\nআইনি নথি:\n{context}\n\nপেশাদারভাবে সঠিক সমাধান দিন:")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", ENHANCED_SYSTEM_PROMPT_EN),
                ("user", "Question/Situation: {question}\n\nLegal Documents:\n{context}\n\nProvide a professional, exact solution:")
            ])
        
        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"question": processed_query, "context": formatted_context})
        
        # Add disclaimer
        if lang == 'bn':
            disclaimer = "\n\n💼 **দাবি পরিহার**: এটি সাধারণ নির্দেশনা। জটিল ক্ষেত্রে আইনজীবীর পরামর্শ নিন।"
        else:
            disclaimer = "\n\n💼 **Disclaimer**: This is general guidance. Consult a lawyer for complex cases."
        response += disclaimer
        
        return response
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Sorry, an error occurred. Please try again. / দুঃখিত, একটি ত্রুটি হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।"

# Main function
def enhanced_retrieve_and_respond(query: str) -> str:
    """Main function for RAG processing"""
    return enhanced_legal_processing(query)

if __name__ == "__main__":
    print("\n🤖 LEGAL BEE - Legal Assistant")
    print("Enter your legal question or situation (or 'quit' to exit):")
    
    while True:
        user_query = input("\n❓ Your Input: ").strip()
        if user_query.lower() in ['quit', 'exit', 'বের হও']:
            print("Thank you for using LEGAL BEE!")
            break
        if user_query:
            print("\n🔍 Processing...")
            response = enhanced_retrieve_and_respond(user_query)
            print(f"\n💡 Response:\n{response}")
        else:
            print("Please enter a valid input.")