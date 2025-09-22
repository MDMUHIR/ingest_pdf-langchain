#!/usr/bin/env python3
"""
ingest_single_en_pdfs.py

Optimized for single English act PDFs.
- Extracts title/act from first ~800 chars to avoid false matches.
- Handles chapter/section persistence per page.
- Assumes no multi-act structure.
- Enriched with translation for bilingual metadata (title-bn, act-number).
- Safe translation with retries and context.
- Chunk saving for long texts, audit logs, OCR fallback.
- Skips short chunks unless key terms like 'Section|Chapter'.
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
DATA_DIR = os.environ.get("DATA_DIR", "data/laws/en_single")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")  # optional
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "ingest_artifacts_en")
CHUNK_TEXT_SAVE_THRESHOLD = int(os.environ.get("CHUNK_TEXT_SAVE_THRESHOLD", 8000))
MIN_CHUNK_LEN = int(os.environ.get("MIN_CHUNK_LEN", 10))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "false").lower() in ("1", "true", "yes")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "chunks"), exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest_en")

# ---------- Utilities ----------
bn_to_en_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", s.replace("\r\n", "\n"))).strip()

def strip_known_footers_and_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"bdlaws\.minlaw\.gov\.bd\S*", "", text, flags=re.I)
    text = re.sub(r"\n?\s*\d+\s*/\s*\d+\s*\n?", "\n", text)
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4}", "", text)  # Dates
    text = "\n".join([ln for ln in text.splitlines() if not re.search(r"copyright|ministry|bdlaws|legislative|parliamentary|affairs|division|\d+/\d+", ln, re.I)])
    return normalize_whitespace(text)

def detect_lang(text: str) -> str:
    return "en"  # Force English for this script

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
                ocr_text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
                if len(ocr_text.strip()) > len(text.strip()):
                    text = strip_known_footers_and_urls(ocr_text)
            yield i+1, normalize_whitespace(text)
        except Exception as e:
            log.exception("Page %d extract failed: %s", i+1, e)
            yield i+1, ""

# ---------- Title & act extraction (English only, limited to first 800 chars, refined patterns) ----------
ENG_TITLE_PATTERNS = [
    re.compile(r'^(The\s+[\w\-\.,\(\) ]+?(?:Act|Code|Ordinance|Rules)[\w \,\:\(\)]*?\d{3,4})', re.I | re.M),
    re.compile(r'^(The\s+[\w\-\.,\(\) ]+?Act,\s*\d{4})', re.I | re.M),
    re.compile(r'^The\s+[\w\-\.,\(\) ]+Act,\s*\d{4}', re.I | re.M),
]
ENG_ACTNO_PATTERNS = [
    re.compile(r'\(ACT NO\.?\s*([IVXLC\d]+)\s*OF\s*(\d{4})\)', re.I),
    re.compile(r'ACT NO\.?\s*([IVXLC\d]+)\s*OF\s*(\d{4})', re.I),
]

def extract_law_title_and_number(path: str, text: str) -> Tuple[str, str]:
    fp = strip_known_footers_and_urls(text or "")[:800]  # Limit to first 800 chars
    for p in ENG_TITLE_PATTERNS:
        m = p.search(fp)
        if m:
            title = m.group(0).strip()
            for a in ENG_ACTNO_PATTERNS:
                am = a.search(fp)
                if am:
                    act = f"ACT NO. {am.group(1)} OF {am.group(2)}"
                    return title, act
            return title, ""
    fname = os.path.basename(path)
    return os.path.splitext(fname)[0].replace("_"," ").replace("-"," "), ""

# ---------- Chapter and Section extraction ----------
ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}  # Add more if needed

def extract_chapter(text: str) -> str:
    chap_num = re.search(r'Chapter\s+(\d+)', text, re.I)
    if chap_num:
        return f"Chapter {chap_num.group(1)}"
    chap_roman = re.search(r'Chapter\s+([IVXLC]+)', text, re.I)
    if chap_roman:
        roman = chap_roman.group(1).upper()
        num = ROMAN_TO_INT.get(roman, roman)
        return f"Chapter {num}"
    return ""

def extract_section(text: str) -> str:
    sec = re.search(r'Section\s+(\d+)', text, re.I)
    if sec:
        return f"Section {sec.group(1)}"
    return ""

# ---------- Splitter ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,
                                          separators=["\n\n", "\n", ".", ";", ":", " "])

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
            title_en, act_en = extract_law_title_and_number(pdf, first_page_text)
            title_bn = safe_translate(title_en, "bn", "Act title")
            title_field = title_en
            act_number_field = f"{safe_translate(act_en, 'bn', 'Act number')} / {act_en}" if act_en else ""
            chunk_count = 0
            for page_no, text in load_pdf_fulltext(pdf):
                if not text.strip(): continue
                current_chapter = extract_chapter(text) or ""
                current_section = extract_section(text) or ""
                for cidx, chunk in enumerate(splitter.split_text(text)):
                    if len(chunk.strip()) < MIN_CHUNK_LEN and not re.search(r'Section|Chapter', chunk, re.I): 
                        continue
                    chunk_section = extract_section(chunk) or current_section
                    chunk_chapter = extract_chapter(chunk) or current_chapter
                    lang_field = "en / bn"
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