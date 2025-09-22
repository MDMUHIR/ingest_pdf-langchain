#!/usr/bin/env python3
"""
ingest_single_bn_pdfs.py

Optimized for single Bengali act PDFs.
- Improved title extraction to handle broken/vertical text.
- Bengali-focused patterns and detection.
"""
import os, glob, hashlib, json, re, time, logging
from typing import List, Dict, Tuple, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClient

# Optional OCR fallback
OCR_AVAILABLE = False
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pass

# ---------- Config ----------
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR", "data/laws/bn_single")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")  # optional
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "ingest_artifacts_bn")
CHUNK_TEXT_SAVE_THRESHOLD = int(os.environ.get("CHUNK_TEXT_SAVE_THRESHOLD", 8000))
MIN_CHUNK_LEN = int(os.environ.get("MIN_CHUNK_LEN", 10))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "false").lower() in ("1", "true", "yes")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "chunks"), exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest_bn")

# ---------- Utilities ----------
bn_to_en_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", s.replace("\r\n", "\n"))).strip()

def strip_known_footers_and_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"bdlaws\.minlaw\.gov\.bd\S*", "", text, flags=re.I)
    text = re.sub(r"\n?\s*\d+\s*/\s*\d+\s*\n?", "\n", text)
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4}", "", text)  # Dates
    # Remove broken vertical lines (single chars)
    text = "\n".join([ln for ln in text.splitlines() if len(ln.strip()) > 2 or re.search(r'[\u0980-\u09FF]', ln)])
    text = "\n".join([ln for ln in text.splitlines() if not re.search(r"copyright|ministry|bdlaws|legislative|parliamentary|affairs|division", ln, re.I)])
    return normalize_whitespace(text)

def detect_lang(text: str) -> str:
    return "bn"  # Force Bengali for this script

def make_doc_id(fname: str, page: int, cidx: int, preview: str) -> str:
    base = f"{fname}|p{page}|c{cidx}|{preview[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

def safe_translate(text: str, target: str, context: str = "", retries: int = 3) -> str:
    if not text.strip() or re.fullmatch(r"[\s\d\W/().,-]+", text):
        return text
    for attempt in range(retries):
        try:
            to_translate = f"{context}: {text}" if context and len(text) > 20 else text
            res = GoogleTranslator(source="auto", target=target).translate(to_translate)
            if context and res.lower().startswith(context.lower()):
                res = re.sub(rf"^{re.escape(context)}\s*[:\-–]\s*", "", res, flags=re.I)
            return res
        except Exception as e:
            log.warning("Translate retry %d failed: %s", attempt+1, e)
            time.sleep(0.3 * (attempt+1))
    return text

# ---------- PDF loader ----------
def load_pdf_fulltext(path: str):
    doc = fitz.open(path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        try:
            text = page.get_text("text") or ""
            text = strip_known_footers_and_urls(text)
            if (len(text.strip()) < 25 or text.count("�")/(len(text)+1) > 0.03) and OCR_AVAILABLE:
                log.info("OCR fallback on %s p%d", os.path.basename(path), i+1)
                pix = page.get_pixmap(dpi=200)
                img_path = os.path.join(ARTIFACTS_DIR, f"ocr_{os.path.basename(path)}_{i+1}.png")
                pix.save(img_path)
                ocr_text = pytesseract.image_to_string(Image.open(img_path), lang="ben")
                if len(ocr_text.strip()) > len(text.strip()):
                    text = strip_known_footers_and_urls(ocr_text)
            yield i+1, normalize_whitespace(text)
        except Exception as e:
            log.exception("Page %d extract failed: %s", i+1, e)
            yield i+1, ""

# ---------- Title & act extraction (Bengali only, improved for broken text) ----------
BN_TITLE_PATTERNS = [
    re.compile(r'^([^\n]+? আইন)[, ]*\d{4}', re.M | re.I),  # e.g., ... আইন, ১৯৯১
    re.compile(r'^(.+?)\s*\(\s*\d{4}\s*সনের\s*\d+\s*নং\s*আইন\s*\)', re.M | re.S),
]
BN_ACTNO_PATTERNS = [
    re.compile(r'\(\s*(\d{4})\s*সনের\s*(\d+)\s*নং\s*আইন\s*\)', re.I),
    re.compile(r'(\d{3,4}\s*সনের\s*\d+\s*নং)', re.I),
    re.compile(r'(\d+\s*নং\s*আইন)', re.I),
]

def extract_law_title_and_number(path: str, text: str) -> Tuple[str, str]:
    fp = strip_known_footers_and_urls(text or "")[:800]  # Limit to first 800 chars
    for p in BN_TITLE_PATTERNS:
        m = p.search(fp)
        if m:
            title = m.group(1).strip()
            for a in BN_ACTNO_PATTERNS:
                am = a.search(fp)
                if am:
                    act = am.group(0).strip()
                    return title, act
            return title, ""
    fname = os.path.basename(path)
    return os.path.splitext(fname)[0].replace("_"," ").replace("-"," "), ""

# ---------- Chapter and Section extraction ----------
def extract_chapter(text: str) -> str:
    chap_bn_num = re.search(r'অধ্যায়\s+([০-৯]+|\d+)', text)
    if chap_bn_num:
        num = chap_bn_num.group(1).translate(bn_to_en_digits)
        return f"অধ্যায় {num}"
    bn_word_to_num = {"প্রথম": "১", "দ্বিতীয়": "২", "তৃতীয়": "৩", "চতুর্থ": "৪", "পঞ্চম": "৫", "ষষ্ঠ": "৬", "সপ্তম": "৭", "অষ্টম": "৮", "নবম": "৯", "দশম": "১০"}
    chap_bn_word = re.search(r'(' + '|'.join(bn_word_to_num.keys()) + r')\s*অধ্যায়', text)
    if chap_bn_word:
        return f"{chap_bn_word.group(1)} অধ্যায়"
    return ""

def extract_section(text: str) -> str:
    sec_bn = re.search(r'ধারা\s+([০-৯]+|\d+)', text)
    if sec_bn:
        num = sec_bn.group(1).translate(bn_to_en_digits)
        return f"{num} / ধারা {num}"
    return ""

# ---------- Splitter ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,
                                          separators=["\n\n", "\n", "।", ".", ";", ":", " "])

# ---------- Embeddings ----------
log.info("Loading embedding model: %s", EMBEDDING_MODEL)
emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------- Pinecone client ----------
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Missing PINECONE_API_KEY / PINECONE_INDEX_NAME")

pc = PineconeClient(api_key=PINECONE_API_KEY)
if PINECONE_ENV:
    pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
idx = pc.Index(PINECONE_INDEX_NAME)

# ---------- Build documents ----------
def build_documents(data_dir: str) -> List[Dict]:
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    log.info("Found %d PDFs in %s", len(pdf_files), data_dir)
    docs, report = [], {}
    for pdf in pdf_files:
        fname = os.path.basename(pdf)
        try:
            first_page_text = next((t for p,t in load_pdf_fulltext(pdf) if p==1), "")
            title_bn, act_bn = extract_law_title_and_number(pdf, first_page_text)
            title_en = safe_translate(title_bn, "en", "Act title")
            title_field = title_bn
            act_number_field = f"{act_bn} / {safe_translate(act_bn, 'en', 'Act number')}" if act_bn else ""
            chunk_count = 0
            for page_no, text in load_pdf_fulltext(pdf):
                if not text.strip(): continue
                current_chapter = extract_chapter(text) or ""
                current_section = extract_section(text) or ""
                for cidx, chunk in enumerate(splitter.split_text(text)):
                    if len(chunk.strip()) < MIN_CHUNK_LEN and not re.search(r'ধারা|অধ্যায়', chunk): 
                        continue
                    chunk_section = extract_section(chunk) or current_section
                    chunk_chapter = extract_chapter(chunk) or current_chapter
                    lang_field = "bn / en"
                    doc_id = make_doc_id(fname, page_no, chunk_count, chunk)
                    metadata = {
                        "doc_id": doc_id, "title": title_field, "title-bn": title_bn, "title-en": title_en,
                        "chapter": chunk_chapter, "section": chunk_section, "act-number": act_number_field,
                        "source": fname, "page": page_no, "lang": lang_field
                    }
                    full_text = chunk.strip()
                    if len(full_text) > CHUNK_TEXT_SAVE_THRESHOLD:
                        path = os.path.join(ARTIFACTS_DIR, "chunks", f"{doc_id}.json")
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump({"doc_id": doc_id, "text": full_text, "metadata": metadata}, fh, ensure_ascii=False)
                        metadata["text_path"] = path
                        metadata["text"] = full_text[:1000] + " ... [TRUNCATED]"
                    else:
                        metadata["text"] = full_text
                    docs.append({"id": doc_id, "text": full_text, "metadata": metadata})
                    chunk_count += 1
            report[fname] = {"chunks": chunk_count, "title": title_field[:200], "act": act_number_field[:200]}
            log.info("Prepared %d chunks from %s", chunk_count, fname)
        except Exception as e:
            log.exception("File %s failed: %s", fname, e)
            report[fname] = {"error": str(e)}
    with open(os.path.join(ARTIFACTS_DIR, "ingest_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info("Total chunks: %d", len(docs))
    return docs

# ---------- Fetch & upsert ----------
# (Same as Script 1)
def fetch_existing_ids(ids: List[str]) -> set:
    found = set()
    for i in range(0, len(ids), 100):
        try:
            resp = idx.fetch(ids=ids[i:i+100])
            vectors = getattr(resp, "vectors", None)
            if vectors: found.update(vectors.keys())
        except Exception as e:
            log.warning("Fetch ids failed: %s", e)
    return found

def upsert_documents_to_pinecone(docs: List[Dict], batch_size=BATCH_SIZE):
    # (Identical to Script 1)
    if not docs: return log.info("No docs to upsert.")
    if SKIP_EXISTING:
        existing = fetch_existing_ids([d["id"] for d in docs])
        docs = [d for d in docs if d["id"] not in existing]
        log.info("Skipping %d existing docs", len(existing))
    if not docs: return log.info("Nothing new to upsert.")
    vec_dim = len(emb.embed_documents([docs[0]["text"]])[0])
    log.info("Embedding dim: %d", vec_dim)
    audit = open(os.path.join(ARTIFACTS_DIR, "upsert_audit.jsonl"), "a", encoding="utf-8")
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            vectors = emb.embed_documents([d["text"] for d in batch])
            payload = [{"id": d["id"], "values": list(map(float, v)), "metadata": d["metadata"]}
                       for d, v in zip(batch, vectors)]
            idx.upsert(vectors=payload)
            for d, v in zip(batch, vectors):
                audit.write(json.dumps({"id": d["id"], "meta": {"src": d["metadata"].get("source"), "page": d["metadata"].get("page")}, "vec_head": list(map(float, v))[:8]}, ensure_ascii=False) + "\n")
            log.info("Upserted %d/%d", min(i + batch_size, len(docs)), len(docs))
        except Exception as e:
            log.exception("Batch %d-%d failed: %s", i, i + batch_size - 1, e)
        time.sleep(0.15)
    audit.close()
    log.info("Audit written.")

# ---------- Main ----------
if __name__ == "__main__":
    docs = build_documents(DATA_DIR)
    upsert_documents_to_pinecone(docs)
    log.info("Done.")