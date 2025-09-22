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
from typing import List, Dict, Tuple, Set, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import logging
from collections import Counter
import numpy as np

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever setup
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb, namespace="")  # Fixed: Default namespace is empty string, not "_default_"

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

# Legal domain keyword extraction (Enriched: Added more domains and keywords for broader coverage)
LEGAL_KEYWORDS_EN = {
    'criminal': ['murder', 'theft', 'fraud', 'assault', 'criminal', 'police', 'fir', 'charge', 'arrest', 'bail', 'trial'],
    'civil': ['contract', 'property', 'land', 'dispute', 'damages', 'civil', 'suit', 'decree', 'injunction', 'specific performance'],
    'family': ['marriage', 'divorce', 'custody', 'alimony', 'family', 'wife', 'husband', 'child', 'adoption', 'inheritance'],
    'banking': ['bank', 'loan', 'defaulter', 'account', 'deposit', 'credit', 'banking', 'foreclosure', 'recovery', 'insolvency'],
    'election': ['election', 'vote', 'candidate', 'officer', 'commission', 'ballot', 'petition', 'disqualification'],
    'labor': ['worker', 'employee', 'salary', 'wage', 'labor', 'employment', 'job', 'termination', 'compensation', 'strike'],
    'cyber': ['internet', 'online', 'digital', 'cyber', 'computer', 'data', 'hacking', 'privacy', 'defamation', 'fraud'],
    'constitutional': ['constitution', 'rights', 'fundamental', 'writ', 'petition', 'supreme court', 'high court'],  # New
    'administrative': ['administrative', 'tribunal', 'service', 'government', 'order', 'appeal'],  # New
    'environmental': ['environment', 'pollution', 'forest', 'wildlife', 'conservation']  # New
}

LEGAL_KEYWORDS_BN = {
    'criminal': ['খুন', 'চুরি', 'প্রতারণা', 'আক্রমণ', 'ফৌজদারি', 'পুলিশ', 'মামলা', 'গ্রেফতার', 'জামিন', 'বিচার'],
    'civil': ['চুক্তি', 'সম্পত্তি', 'জমি', 'বিরোধ', 'ক্ষতিপূরণ', 'দেওয়ানি', 'মামলা', 'ডিক্রি', 'নিষেধাজ্ঞা', 'নির্দিষ্ট কর্মসম্পাদন'],
    'family': ['বিবাহ', 'তালাক', 'অভিভাবকত্ব', 'ভরণপোষণ', 'পারিবারিক', 'স্ত্রী', 'স্বামী', 'সন্তান', 'দত্তক', 'উত্তরাধিকার'],
    'banking': ['ব্যাংক', 'ঋণ', 'খেলাপি', 'অ্যাকাউন্ট', 'আমানত', 'ঋণ', 'ব্যাংকিং', 'ফোরক্লোজার', 'আদায়', 'দেউলিয়া'],
    'election': ['নির্বাচন', 'ভোট', 'প্রার্থী', 'কর্মকর্তা', 'কমিশন', 'ব্যালট', 'পিটিশন', 'অযোগ্যতা'],
    'labor': ['শ্রমিক', 'কর্মচারী', 'বেতন', 'মজুরি', 'শ্রম', 'চাকরি', 'কাজ', 'বরখাস্ত', 'ক্ষতিপূরণ', 'ধর্মঘট'],
    'cyber': ['ইন্টারনেট', 'অনলাইন', 'ডিজিটাল', 'সাইবার', 'কম্পিউটার', 'তথ্য', 'হ্যাকিং', 'গোপনীয়তা', 'মানহানি', 'প্রতারণা'],
    'constitutional': ['সংবিধান', 'অধিকার', 'মৌলিক', 'রিট', 'পিটিশন', 'সুপ্রিম কোর্ট', 'হাই কোর্ট'],  # New
    'administrative': ['প্রশাসনিক', 'ট্রাইব্যুনাল', 'সার্ভিস', 'সরকার', 'আদেশ', 'আপিল'],  # New
    'environmental': ['পরিবেশ', 'দূষণ', 'বন', 'বন্যপ্রাণী', 'সংরক্ষণ']  # New
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
                expansions.extend(LEGAL_KEYWORDS_BN.get(domain, [])[:5])  # Enriched: Increased to top 5 for better coverage
            else:
                expansions.extend(LEGAL_KEYWORDS_EN.get(domain, [])[:5])
    
    return list(set(expansions))

# Enhanced translation with legal context
ENHANCED_TRANSLATE_TO_ENGLISH_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal translation expert. Translate this Bengali legal query to English while preserving all legal nuances and terminology.

Important guidelines:
- Maintain exact legal terms and act names
- Preserve section numbers and references
- Keep the formal legal tone
- Use standard legal English terminology
- Ensure the translation captures the legal intent

Bengali Query: {query}
Legal Domain: {domain}

Precise English Translation:"""
)

ENHANCED_TRANSLATE_TO_BENGALI_PROMPT = ChatPromptTemplate.from_template(
    """You are a legal translation expert. Translate this English legal query to Bengali while preserving all legal nuances and terminology.

Important guidelines:
- Maintain exact legal terms and act names in Bengali or keep English terms where standard
- Preserve section numbers and references
- Keep the formal legal tone
- Use standard Bengali legal terminology
- Ensure the translation captures the legal intent

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

# Multi-strategy retrieval system (Enriched: Use similarity_search_with_score for actual relevance scores; added min_score filter)
class MultiStrategyRetriever:
    def __init__(self, vectorstore, k=30, min_score=0.6):
        self.vectorstore = vectorstore
        self.k = k
        self.min_score = min_score  # Enriched: Minimum cosine similarity threshold
    
    def hybrid_search(self, query: str, lang: str) -> List[Document]:
        """Hybrid search combining multiple strategies with actual scores"""
        all_results = []  # Store (doc, score) pairs
        
        # Helper function to perform search and filter by min_score
        def perform_search(search_query: str, strategy: str, score_multiplier: float = 1.0):
            try:
                results = self.vectorstore.similarity_search_with_score(search_query, k=self.k)
                filtered = [(doc, score * score_multiplier) for doc, score in results if score >= self.min_score]
                for doc, score in filtered:
                    doc.metadata['retrieval_strategy'] = strategy
                    doc.metadata['relevance_score'] = score
                return [doc for doc, _ in filtered]
            except Exception as e:
                logger.error(f"Search for '{search_query}' failed: {e}")
                return []
        
        # Strategy 1: Direct similarity search
        direct_docs = perform_search(query, 'direct', 1.0)
        all_results.extend(direct_docs)
        
        # Strategy 2: Domain-specific search (slightly penalized score)
        domains = extract_legal_domain(query, lang)
        for domain in domains:
            domain_keywords = LEGAL_KEYWORDS_BN.get(domain, []) if lang == 'bn' else LEGAL_KEYWORDS_EN.get(domain, [])
            for keyword in domain_keywords[:3]:  # Limit to top 3 keywords
                keyword_docs = perform_search(keyword, f'domain_{domain}', 0.8)  # Enriched: Multiplier for domain strategy
                all_results.extend(keyword_docs)
        
        # Strategy 3: Translated query search
        target_lang = 'en' if lang == 'bn' else 'bn'
        translated_query = enhanced_translate_query(query, lang, target_lang)
        if translated_query != query:
            translated_docs = perform_search(translated_query, 'translated', 0.95)  # Enriched: High multiplier for translated
            all_results.extend(translated_docs)
        
        # Strategy 4: Expanded query search (more penalized)
        expanded_terms = expand_query_terms(query, lang)
        for term in expanded_terms[:5]:  # Limit to top 5 expanded terms
            expanded_docs = perform_search(term, 'expanded', 0.7)  # Enriched: Multiplier for expanded
            all_results.extend(expanded_docs)
        
        return all_results  # Now list of docs with scores in metadata
    
    def intelligent_deduplication(self, docs: List[Document]) -> List[Document]:
        """Intelligent deduplication with relevance scoring (Enriched: Aggregate max score for duplicates)"""
        unique_docs = {}
        
        for doc in docs:
            # Create composite key
            doc_id = doc.metadata.get('doc_id')
            page = doc.metadata.get('page', 0)
            section = doc.metadata.get('section', '')
            act_num = doc.metadata.get('act-number', '')
            
            key = f"{doc_id}_{act_num}_{page}_{section}" if doc_id else f"{hash(doc.page_content[:200])}_{page}_{section}"
            
            if key not in unique_docs or doc.metadata.get('relevance_score', 0) > unique_docs[key].metadata.get('relevance_score', 0):
                unique_docs[key] = doc  # Keep the one with highest score
        
        # Sort by relevance score and retrieval strategy priority
        strategy_priority = {
            'direct': 4,
            'translated': 3,
            'domain_': 2,  # Prefix matching for domain strategies
            'expanded': 1
        }
        
        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda x: (
                x.metadata.get('relevance_score', 0),
                max([v for k, v in strategy_priority.items() if x.metadata.get('retrieval_strategy', '').startswith(k)] + [0])
            ),
            reverse=True
        )
        
        return sorted_docs[:25]  # Return top 25 most relevant documents

# Initialize enhanced retriever (Enriched: Added min_score parameter)
multi_retriever = MultiStrategyRetriever(vectorstore, min_score=0.65)  # Enriched: Slightly higher threshold for quality

# Enhanced document relevance scoring (Enriched: Combined with vector score if available)
def score_document_relevance(doc: Document, query: str, lang: str) -> float:
    """Score document relevance based on multiple factors"""
    score = doc.metadata.get('relevance_score', 0.5) * 0.5  # Start with 50% weight from vector score
    content = doc.page_content.lower()
    query_lower = query.lower()
    
    # Direct query term matching
    query_words = set(query_lower.split())
    content_words = set(content.split())
    common_words = query_words.intersection(content_words)
    if query_words:
        score += (len(common_words) / len(query_words)) * 0.2  # Reduced weight
    
    # Legal domain relevance
    domains = extract_legal_domain(query, lang)
    for domain in domains:
        keywords = LEGAL_KEYWORDS_BN.get(domain, []) if lang == 'bn' else LEGAL_KEYWORDS_EN.get(domain, [])
        domain_matches = sum(1 for keyword in keywords if keyword in content)
        if keywords:
            score += (domain_matches / len(keywords)) * 0.2  # Reduced weight
    
    # Metadata quality scoring
    metadata = doc.metadata
    if metadata.get('section'):
        score += 0.05
    if metadata.get('act-number'):
        score += 0.05
    if metadata.get('title'):
        score += 0.05
    
    return min(score, 1.0)

# Enhanced context formatting with intelligent ranking
def format_context_intelligently(docs: List[Document], query: str, lang: str) -> str:
    """Format context with intelligent ranking and relevance scoring"""
    if not docs:
        return "No relevant documents found."
    
    # Score and rank documents
    scored_docs = []
    for doc in docs:
        relevance_score = score_document_relevance(doc, query, lang)
        doc.metadata['computed_relevance'] = relevance_score
        scored_docs.append((doc, relevance_score))
    
    # Sort by computed relevance
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Group by act/law for better organization
    grouped_docs = {}
    for doc, score in scored_docs[:20]:  # Top 20 most relevant
        act_key = doc.metadata.get('act-number', 'Unknown Act')
        if act_key not in grouped_docs:
            grouped_docs[act_key] = []
        grouped_docs[act_key].append((doc, score))
    
    # Format grouped documents
    formatted_sections = []
    for act_key, doc_list in grouped_docs.items():
        # Sort documents within each act by relevance
        doc_list.sort(key=lambda x: x[1], reverse=True)
        
        act_sections = []
        for doc, score in doc_list[:5]:  # Top 5 per act
            meta = doc.metadata
            
            # Get title in appropriate language
            if lang == 'bn':
                title = meta.get('title-bn', meta.get('title', 'Unknown'))
            else:
                title = meta.get('title-en', meta.get('title', 'Unknown'))
            
            # Format citation
            section = meta.get('section', '')
            page = meta.get('page', '?')
            
            citation_parts = [title]
            if act_key and act_key != 'Unknown Act':
                citation_parts.append(act_key)
            if section:
                citation_parts.append(f"Section {section}")
            citation_parts.append(f"p.{page}")
            citation_parts.append(f"(Relevance: {score:.2f})")
            
            cite = " - ".join(citation_parts)
            
            # Clean and truncate content (Shortened: Reduced from 800 to 400 for brevity)
            content = doc.page_content.strip()
            if len(content) > 400:
                content = content[:400] + "..."
            
            act_sections.append(f"[{cite}]\n{content}")
        
        if act_sections:
            formatted_sections.append("\n\n".join(act_sections))
    
    return "\n\n--- NEXT ACT ---\n\n".join(formatted_sections)

# Enhanced system prompts with better legal analysis and conciseness instruction
ENHANCED_SYSTEM_PROMPT_EN = """You are LEGAL BEE, an expert AI Legal Assistant specializing in Bangladeshi Law with advanced legal analysis capabilities.

### Core Expertise:
You have deep knowledge of Bangladeshi legal system including constitutional law, civil law, criminal law, family law, banking law, labor law, cyber law, and administrative law. You understand legal precedents, procedural requirements, and practical implications.

### Response Structure (MANDATORY):
Always structure your response with these exact sections, keeping each section concise and focused on the main topic:

**🎯 LEGAL ISSUE IDENTIFICATION**
- Briefly identify the specific legal problem(s)
- Categorize the area of law involved
- Highlight urgency level (if applicable)

**📚 APPLICABLE LAW & ANALYSIS**
- State key relevant laws, acts, and sections with precise citations
- Concisely explain what each law means
- Briefly analyze application to the situation

**⚖️ LEGAL RIGHTS & REMEDIES**
- List main rights of the person
- Key available legal remedies
- Main potential outcomes

**📋 STEP-BY-STEP ACTION PLAN**
- Key immediate actions
- Essential documentation
- Main legal procedures
- Critical timeline considerations

**⚠️ IMPORTANT CONSIDERATIONS**
- Main challenges
- Key time limitations
- Basic costs/practical notes
- When to seek help

### Quality Standards:
- Be concise: Focus on core concepts and main points only
- Limit explanations to essentials
- Base EVERY legal statement on retrieved documents with proper citations [Title - Act Number - Section (p.Page)]
- Provide specific section numbers, not general references
- Explain legal concepts in plain English while maintaining accuracy
- Give practical, actionable advice within Bangladeshi legal framework
- Address key counterarguments briefly

### Limitations:
- State clearly when information is insufficient
- Recommend consultation with qualified lawyers
- Never guarantee legal outcomes
- Stay strictly within Bangladeshi law"""

ENHANCED_SYSTEM_PROMPT_BN = """আপনি LEGAL BEE, বাংলাদেশী আইনে বিশেষজ্ঞ একজন দক্ষ AI আইনি সহায়ক যিনি উন্নত আইনি বিশ্লেষণের ক্ষমতা রাখেন।

### মূল দক্ষতা:
আপনার বাংলাদেশী আইনি ব্যবস্থার গভীর জ্ঞান আছে যার মধ্যে রয়েছে সাংবিধানিক আইন, দেওয়ানি আইন, ফৌজদারি আইন, পারিবারিক আইন, ব্যাংকিং আইন, শ্রম আইন, সাইবার আইন এবং প্রশাসনিক আইন। আপনি আইনি নজির, প্রক্রিয়াগত প্রয়োজনীয়তা এবং ব্যবহারিক প্রভাব বোঝেন।

### উত্তরের কাঠামো (বাধ্যতামূলক):
সর্বদা এই নির্দিষ্ট বিভাগগুলি দিয়ে আপনার উত্তর সাজান, প্রত্যেক বিভাগকে সংক্ষিপ্ত রাখুন এবং মূল বিষয়ে ফোকাস করুন:

**🎯 আইনি সমস্যা চিহ্নিতকরণ**
- সংক্ষেপে নির্দিষ্ট আইনি সমস্যা(গুলি) চিহ্নিত করুন
- সংশ্লিষ্ট আইনের ক্ষেত্র শ্রেণীবদ্ধ করুন
- জরুরী মাত্রা তুলে ধরুন (যদি প্রযোজ্য হয়)

**📚 প্রযোজ্য আইন ও বিশ্লেষণ**
- সুনির্দিষ্ট উদ্ধৃতি সহ মূল প্রাসঙ্গিক আইন, অ্যাক্ট এবং ধারাসমূহ উল্লেখ করুন
- সংক্ষেপে প্রতিটি আইনের অর্থ ব্যাখ্যা করুন
- সংক্ষেপে পরিস্থিতিতে প্রয়োগ বিশ্লেষণ করুন

**⚖️ আইনি অধিকার ও প্রতিকার**
- ব্যক্তির মূল অধিকারসমূহ তালিকাভুক্ত করুন
- মূল উপলব্ধ আইনি প্রতিকার
- মূল সম্ভাব্য ফলাফল

**📋 ধাপে ধাপে কর্মপরিকল্পনা**
- মূল তাৎক্ষণিক পদক্ষেপ
- অত্যাবশ্যক কাগজপত্র
- মূল আইনি প্রক্রিয়া
- গুরুত্বপূর্ণ সময়ের বিবেচনা

**⚠️ গুরুত্বপূর্ণ বিবেচনা**
- মূল চ্যালেঞ্জ
- মূল সময়সীমা
- মৌলিক খরচ/ব্যবহারিক নোট
- সাহায্য নেওয়ার সময়

### গুণমানের মান:
- সংক্ষিপ্ত হোন: মূল ধারণা এবং প্রধান পয়েন্টগুলিতে ফোকাস করুন
- ব্যাখ্যা অপরিহার্যতায় সীমাবদ্ধ রাখুন
- প্রতিটি আইনি বক্তব্য উদ্ধৃত নথিপত্রের উপর ভিত্তি করুন সঠিক উদ্ধৃতি সহ [শিরোনাম - আইন নম্বর - ধারা (পৃ.পেজ)]
- সাধারণ রেফারেন্স নয়, নির্দিষ্ট ধারা নম্বর প্রদান করুন
- নির্ভুলতা বজায় রেখে সহজ বাংলায় আইনি ধারণা ব্যাখ্যা করুন
- বাংলাদেশী আইনি কাঠামোর মধ্যে ব্যবহারিক, কার্যকর পরামর্শ দিন
- মূল পাল্টা যুক্তি সংক্ষেপে সমাধান করুন

### সীমাবদ্ধতা:
- তথ্য অপর্যাপ্ত হলে স্পষ্টভাবে উল্লেখ করুন
- যোগ্য আইনজীবীদের পরামর্শের সুপারিশ করুন
- আইনি ফলাফলের গ্যারান্টি দেবেন না
- কঠোরভাবে বাংলাদেশী আইনের মধ্যে সীমাবদ্ধ থাকুন"""

# Enhanced fact extraction with legal categorization
ENHANCED_FACT_EXTRACT_PROMPT_EN = ChatPromptTemplate.from_template(
    """As a legal expert, extract and categorize the key facts from this personal legal situation. Focus on legally relevant elements that would be important for case analysis. Be concise.

Personal Story: {query}

Provide brief:
1. KEY LEGAL FACTS
2. PARTIES INVOLVED
3. LEGAL ISSUES
4. RELEVANT DATES
5. EVIDENCE
6. DESIRED OUTCOME

Extracted Legal Analysis:"""
)

ENHANCED_FACT_EXTRACT_PROMPT_BN = ChatPromptTemplate.from_template(
    """একজন আইনি বিশেষজ্ঞ হিসেবে, এই ব্যক্তিগত আইনি পরিস্থিতি থেকে মূল তথ্যগুলি বের করুন এবং শ্রেণীবদ্ধ করুন। মামলা বিশ্লেষণের জন্য গুরুত্বপূর্ণ আইনিভাবে প্রাসঙ্গিক উপাদানগুলিতে মনোনিবেশ করুন। সংক্ষিপ্ত হোন।

ব্যক্তিগত ঘটনা: {query}

সংক্ষিপ্তভাবে প্রদান করুন:
১. মূল আইনি তথ্য
২. জড়িত পক্ষসমূহ
৩. আইনি সমস্যা
৪. প্রাসঙ্গিক তারিখ
৫. প্রমাণ
৬. কাঙ্ক্ষিত ফলাফল

নিষ্কাশিত আইনি বিশ্লেষণ:"""
)

# Advanced query processing with legal intelligence
def process_query_with_legal_intelligence(query: str) -> Dict[str, Any]:
    """Advanced query processing with legal domain intelligence"""
    lang, confidence = detect_language_with_confidence(query)
    domains = extract_legal_domain(query, lang)
    
    # Classify query complexity
    complexity_indicators = [
        'multiple parties', 'several issues', 'complex', 'বিভিন্ন পক্ষ', 'জটিল',
        'urgent', 'জরুরি', 'immediate', 'তাৎক্ষণিক'
    ]
    is_complex = any(indicator in query.lower() for indicator in complexity_indicators)
    
    # Detect urgency
    urgency_indicators = [
        'urgent', 'emergency', 'immediate', 'জরুরি', 'তাৎক্ষণিক', 'এখনই'
    ]
    is_urgent = any(indicator in query.lower() for indicator in urgency_indicators)
    
    return {
        'language': lang,
        'confidence': confidence,
        'domains': domains,
        'is_complex': is_complex,
        'is_urgent': is_urgent,
        'expanded_terms': expand_query_terms(query, lang)
    }

# Enhanced main processing function (Enriched: Added try-except in more places for robustness)
def enhanced_legal_processing(query: str) -> str:
    """Enhanced legal processing with multi-step analysis"""
    try:
        # Step 1: Intelligent query analysis
        query_analysis = process_query_with_legal_intelligence(query)
        lang = query_analysis['language']
        
        logger.info(f"Query Analysis: {query_analysis}")
        
        # Step 2: Classify query type with enhanced logic
        classifier_prompt = ChatPromptTemplate.from_template(
            """Analyze this legal query and classify it as:
            - 'story': Personal legal situation requiring fact extraction and analysis
            - 'direct': Direct legal question about laws, procedures, or rights
            - 'research': Research question about legal concepts or comparative analysis
            
            Consider the language, context, and complexity.
            Query: {query}
            Domains: {domains}
            
            Classification (one word only):"""
        )
        
        classifier_chain = classifier_prompt | llm | StrOutputParser()
        query_type = classifier_chain.invoke({
            "query": query,
            "domains": ", ".join(query_analysis['domains'])
        }).strip().lower()
        
        logger.info(f"Query type: {query_type}")
        
        # Step 3: Enhanced fact extraction for story-type queries
        processed_query = query
        if query_type == 'story':
            try:
                if lang == 'bn':
                    fact_chain = ENHANCED_FACT_EXTRACT_PROMPT_BN | llm | StrOutputParser()
                else:
                    fact_chain = ENHANCED_FACT_EXTRACT_PROMPT_EN | llm | StrOutputParser()
                
                extracted_facts = fact_chain.invoke({"query": query})
                
                # Combine extracted facts with original query
                if lang == 'bn':
                    processed_query = f"আইনি বিশ্লেষণ: {extracted_facts}\n\nমূল প্রশ্ন: {query}"
                else:
                    processed_query = f"Legal Analysis: {extracted_facts}\n\nOriginal Query: {query}"
            except Exception as e:
                logger.warning(f"Fact extraction failed: {e}")
        
        # Step 4: Multi-strategy document retrieval
        retrieved_docs = multi_retriever.hybrid_search(processed_query, lang)
        logger.info(f"Retrieved {len(retrieved_docs)} documents from hybrid search")
        
        # Step 5: Intelligent deduplication and ranking
        final_docs = multi_retriever.intelligent_deduplication(retrieved_docs)
        logger.info(f"Final documents after deduplication: {len(final_docs)}")
        
        # Step 6: Enhanced context formatting
        formatted_context = format_context_intelligently(final_docs, processed_query, lang)
        
        # Step 7: Generate enhanced response
        if lang == 'bn':
            prompt = ChatPromptTemplate.from_messages([
                ("system", ENHANCED_SYSTEM_PROMPT_BN),
                ("user", "প্রশ্ন: {question}\n\nআইনি নথি ও প্রসঙ্গ:\n{context}\n\nসংক্ষিপ্তভাবে মূল বিষয়ে ফোকাস করে উত্তর দিন:")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", ENHANCED_SYSTEM_PROMPT_EN),
                ("user", "Question: {question}\n\nLegal Documents & Context:\n{context}\n\nRespond concisely focusing on the main topic:")
            ])
        
        # Step 8: Generate response with enhanced chain
        rag_chain = prompt | llm | StrOutputParser()
        
        response = rag_chain.invoke({
            "question": query,
            "context": formatted_context
        })
        
        # Step 9: Post-process response for quality enhancement
        enhanced_response = post_process_response(response, query_analysis, final_docs)
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error in enhanced_legal_processing: {e}")
        error_msg = (
            f"দুঃখিত, আপনার প্রশ্ন প্রক্রিয়াকরণে সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।\nত্রুটি: {str(e)}"
            if detect_language_with_confidence(query)[0] == 'bn' else
            f"Sorry, there was an error processing your query. Please try again.\nError: {str(e)}"
        )
        return error_msg

def post_process_response(response: str, query_analysis: Dict, docs: List[Document]) -> str:
    """Post-process response to add quality enhancements"""
    lang = query_analysis['language']
    
    # Add document summary if many documents were used
    if len(docs) > 10:
        if lang == 'bn':
            doc_summary = f"\n\n📊 **নথি সারাংশ**: এই উত্তরটি {len(docs)}টি আইনি নথি থেকে প্রাপ্ত তথ্যের ভিত্তিতে তৈরি করা হয়েছে।"
        else:
            doc_summary = f"\n\n📊 **Document Summary**: This response is based on analysis of {len(docs)} legal documents."
        response += doc_summary
    
    # Add urgency notice if urgent query detected
    if query_analysis.get('is_urgent'):
        if lang == 'bn':
            urgency_notice = "\n\n🚨 **জরুরি নোটিশ**: এটি একটি জরুরি আইনি বিষয় বলে মনে হচ্ছে। অবিলম্বে একজন যোগ্য আইনজীবীর সাথে যোগাযোগ করুন।"
        else:
            urgency_notice = "\n\n🚨 **Urgent Notice**: This appears to be an urgent legal matter. Please contact a qualified lawyer immediately."
        response += urgency_notice
    
    # Add complexity warning if complex query detected
    if query_analysis.get('is_complex'):
        if lang == 'bn':
            complexity_warning = "\n\n⚠️ **জটিলতার সতর্কতা**: এটি একটি জটিল আইনি বিষয়। বিশেষজ্ঞ আইনি পরামর্শ অত্যন্ত সুপারিশ করা হয়।"
        else:
            complexity_warning = "\n\n⚠️ **Complexity Warning**: This is a complex legal matter. Expert legal consultation is highly recommended."
        response += complexity_warning
    
    # Add disclaimer
    if lang == 'bn':
        disclaimer = "\n\n💼 **দাবি পরিহার**: এই তথ্য শুধুমাত্র সাধারণ নির্দেশনার জন্য। নির্দিষ্ট আইনি পরামর্শের জন্য একজন লাইসেন্সপ্রাপ্ত আইনজীবীর সাথে পরামর্শ করুন।"
    else:
        disclaimer = "\n\n💼 **Disclaimer**: This information is for general guidance only. Consult with a licensed lawyer for specific legal advice."
    response += disclaimer
    
    return response

# Enhanced retrieval function for backward compatibility
def enhanced_retrieve_and_respond(query: str) -> str:
    """Main function for enhanced RAG processing"""
    return enhanced_legal_processing(query)

# Quality assessment function (Enriched: More metrics)
def assess_response_quality(response: str, query: str, docs: List[Document]) -> Dict[str, Any]:
    """Assess the quality of generated response"""
    # Check citation coverage
    citation_count = len(re.findall(r'\[.*?\]', response))
    
    # Check structure completeness
    required_sections_en = ['LEGAL ISSUE', 'APPLICABLE LAW', 'RIGHTS & REMEDIES', 'ACTION PLAN', 'CONSIDERATIONS']
    required_sections_bn = ['আইনি সমস্যা', 'প্রযোজ্য আইন', 'অধিকার ও প্রতিকার', 'কর্মপরিকল্পনা', 'বিবেচনা']
    
    lang = detect_language_with_confidence(query)[0]
    required_sections = required_sections_bn if lang == 'bn' else required_sections_en
    
    sections_found = sum(1 for section in required_sections if section in response.upper())
    structure_completeness = sections_found / len(required_sections)
    
    # Enriched: Add average relevance score
    avg_relevance = np.mean([doc.metadata.get('computed_relevance', 0) for doc in docs]) if docs else 0
    
    return {
        'citation_count': citation_count,
        'document_count': len(docs),
        'structure_completeness': structure_completeness,
        'response_length': len(response.split()),
        'average_relevance': avg_relevance,
        'language': lang
    }

# Test function with quality assessment
def comprehensive_test():
    """Comprehensive test with quality assessment"""
    test_queries = [
        {
            'query': "Under the Bank Companies Act, 1991, what legal actions can be taken if a bank refuses to provide loan documents to a borrower?",
            
        },
        {
            'query': "Under the Election Officers (Special Provisions) Act, 1991 of Bangladesh, what legal actions can be taken if an election officer is found to be biased?",
            
        },
       
        {
            'query': "বাংলাদেশের নির্বাচন কর্মকর্তা (বিশেষ বিধান) আইন, ১৯৯১ অনুসারে, যদি কোনও নির্বাচন কর্মকর্তা পক্ষপাতদুষ্ট হন তবে কী কী আইনি ব্যবস্থা নেওয়া যেতে পারে?",
            
        }
    ]
    
    print("🔍 COMPREHENSIVE RAG SYSTEM TEST")
    print("=" * 60)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        print(f"\n🧪 TEST CASE {i}")
        print(f"Query: {query}")
        print("-" * 50)
        
        # Process query
        response = enhanced_retrieve_and_respond(query)
        
        # Assess quality (Simulate docs for test; in real, pass actual docs)
        quality_score = assess_response_quality(response, query, [])  # Placeholder docs
        
        print(f"📊 Response Quality:")
        print(f"   - Length: {quality_score['response_length']} words")
        print(f"   - Citation Count: {quality_score['citation_count']}")
        print(f"   - Structure Completeness: {quality_score['structure_completeness'] * 100:.1f}%")
        print(f"   - Average Relevance: {quality_score['average_relevance']:.2f}")
        
        print(f"\n💡 RESPONSE:")
        print(response)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    # Run comprehensive test
    # comprehensive_test()
    
    # Interactive mode
    print("\n🤖 LEGAL BEE - Enhanced RAG System")
    print("Type your legal question (or 'quit' to exit):")
    
    while True:
        user_query = input("\n❓ Your Question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'বের হও', 'চলে যাও']:
            print("Thank you for using LEGAL BEE! / LEGAL BEE ব্যবহার করার জন্য ধন্যবাদ!")
            break
        
        if user_query:
            print("\n🔍 Processing your query...")
            response = enhanced_retrieve_and_respond(user_query)
            print(f"\n💡 LEGAL BEE's Response:\n{response}")
        else:
            print("Please enter a valid question.")