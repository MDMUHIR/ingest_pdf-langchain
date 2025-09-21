import os, glob, hashlib, re, time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from deep_translator import GoogleTranslator

load_dotenv()
DATA_DIR = "data/laws"

# ---------------- Utils ----------------
bn_to_en_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ---------------- PDF Loader ----------------
def load_pdf_fulltext(path):
    """Extracts page-wise text from PDF without losing Unicode chars."""
    doc = fitz.open(path)
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        yield page_no + 1, text.strip()

# ---------------- Text Splitter ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", "।", ".", ";", ":", " "],
)

# ---------------- Embeddings (FREE) ----------------
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

# ---------------- Vector Store ----------------
index_name = os.environ["PINECONE_INDEX_NAME"]
api_key = os.environ["PINECONE_API_KEY"]
from pinecone import Pinecone
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb)

# ---------------- Helpers ----------------
def detect_lang(text: str) -> str:
    return "bn" if re.search(r"[\u0980-\u09FF]", text) else "en"

def make_doc_id(content, meta):
    h = hashlib.sha256((content + str(meta)).encode("utf-8")).hexdigest()
    return h[:32]

def extract_law_title_and_number(pdf_path, first_page_text):
    filename = os.path.basename(pdf_path).split('.')[0]
    title = None
    act_no = None
    
    # Exclude date patterns
    first_page_text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '', first_page_text).strip()
    
    # English patterns
    english_title_pattern = re.compile(r'(The\s+[A-Za-z\s,]+(?:Act|Code),?\s+\d{4})', re.IGNORECASE)
    english_act_no_pattern = re.compile(r'\(Act\s+No\.\s+([IVXCLD]+)\s+of\s+\d{4}\)', re.IGNORECASE)
    
    # Bangla patterns
    bangla_title_pattern = re.compile(r'(.+?আইন),?\s*(\d{4}|\d+ নং আইন হইতে \d+ নং আইন পর্যন্ত)', re.DOTALL)
    bangla_act_no_pattern = re.compile(r'\((\d+ নং আইন)\)')
    
    # Check English
    match = english_title_pattern.search(first_page_text)
    if match:
        title = match.group(1).strip()
        act_match = english_act_no_pattern.search(first_page_text)
        act_no = act_match.group(1).strip() if act_match else None
    else:
        # Check Bangla
        match = bangla_title_pattern.search(first_page_text)
        if match:
            title_base = match.group(1).strip()
            year_or_range = match.group(2).strip()
            title = f"{title_base}, {year_or_range}"
            act_match = bangla_act_no_pattern.search(first_page_text)
            act_no = act_match.group(1).strip() if act_match else None
    
    if not title:
        title = filename.replace('-', ' ').replace('_', ' ').title()
    
    return title, act_no

def translate_text(text, target_lang, context="legal act"):
    if not text or not text.strip():
        return text
    try:
        # Extract year or range
        year_match = re.search(r',\s*(\d{4}|\d+ নং আইন হইতে \d+ নং আইন পর্যন্ত)$', text)
        if year_match:
            text_base = text[:year_match.start()].strip()
            year_part = year_match.group(1)
            # Convert Bangla digits to English in year_part if present
            year_part = year_part.translate(bn_to_en_digits) if target_lang == "en" else year_part
        else:
            text_base = text
            year_part = None
        
        # Add context to improve translation
        trans = GoogleTranslator(source='auto', target=target_lang)
        translated = trans.translate(f"{context}: {text_base}")
        time.sleep(0.5)  # Rate limit delay
        
        # Reattach year in appropriate format
        if year_part:
            if target_lang == "en":
                return f"{translated}, {year_part}" if "from" not in year_part else translated.replace("from", f"from {year_part}")
            else:
                return f"{translated}, {year_part.translate(bn_to_en_digits) if re.search(r'[\u0980-\u09FF]', year_part) else year_part}"
        return translated
    except Exception as e:
        print(f"Translation error for '{text}' to {target_lang}: {e}")
        return text  # Fallback to original if translation fails

# ---------------- Prepare Documents ----------------
all_docs = []
for pdf in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
    first_page_text = next((text for page_no, text in load_pdf_fulltext(pdf) if page_no == 1), "")
    original_title, act_no = extract_law_title_and_number(pdf, first_page_text)
    print(f"Processing: {os.path.basename(pdf)}")
    print(f"Extracted title: {original_title}, Act No.: {act_no}")
    
    title_lang = detect_lang(original_title)
    if title_lang == "bn":
        title_bn = original_title
        title_en = translate_text(original_title, "en", "legal act")
    else:
        title_en = original_title
        title_bn = translate_text(original_title, "bn", "আইন সংক্রান্ত কাগজ")  # Improved context
    
    print(f"Translated - BN: {title_bn}, EN: {title_en}")
    
    doc_gen = load_pdf_fulltext(pdf)
    for page, text in doc_gen:
        if not text or len(text.strip()) < 15:
            continue
        
        section_match = re.search(r'Section\s+(\d+)', text, re.IGNORECASE)
        chapter_match = re.search(r'Chapter\s+(\w+)', text, re.IGNORECASE)
        bn_section_match = re.search(r'ধারা\s+(\d+)', text)
        bn_chapter_match = re.search(r'অধ্যায়\s+(\w+)', text)
        
        section = f"Section {section_match.group(1)}" if section_match else (f"ধারা {bn_section_match.group(1)}" if bn_section_match else None)
        chapter = f"Chapter {chapter_match.group(1)}" if chapter_match else (f"অধ্যায় {bn_chapter_match.group(1)}" if bn_chapter_match else None)
        
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source_path": pdf,
                "title_bn": title_bn,
                "title_en": title_en,
                "act_no": act_no,
                "page": page,
                "chunk": i,
                "lang": detect_lang(chunk),
                "section": section,
                "chapter": chapter
            }
            all_docs.append(Document(page_content=chunk, metadata=metadata))

print(f"Prepared {len(all_docs)} chunks from {len(glob.glob(os.path.join(DATA_DIR, '*.pdf')))} files.")

# ---------------- Upload to Pinecone ----------------
BATCH = 64
filtered_docs = [doc for doc in all_docs if doc.page_content.strip()]
print(f"Filtered to {len(filtered_docs)} non-empty chunks from {len(all_docs)} total chunks")

for i in range(0, len(filtered_docs), BATCH):
    try:
        batch = filtered_docs[i:i + BATCH]
        batch_ids = []
        for doc in batch:
            metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            doc.metadata = metadata
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source_path", "unknown"))
            doc_id = make_doc_id(doc.page_content, doc.metadata)
            doc.metadata["doc_id"] = doc_id
            batch_ids.append(doc_id)
        
        vectorstore.add_documents(documents=batch, ids=batch_ids)
        print(f"Upserted {i + len(batch)}/{len(filtered_docs)}")
    except Exception as e:
        print(f"Error uploading batch {i // BATCH + 1}: {str(e)}")
        continue
    
    time.sleep(1)