#!/usr/bin/env python3
"""
ingest_pdfs_to_pinecone.py

- Reads PDFs from DATA_DIR
- Extracts titles / act numbers (bn/en)
- Splits pages into chunks
- Computes embeddings with HuggingFace LaBSE (or other model)
- Upserts to Pinecone as objects: {"id":..., "values":[...], "metadata":{...}}
- Saves an ingestion audit file + chunk text files (to avoid losing full text)

Notes:
- This script does NOT create indexes. The index must already exist.
- Optional env var SKIP_EXISTING=true will check Pinecone for existing doc_ids and skip them.
"""
import os
import glob
import hashlib
import json
import re
import time
import logging
from typing import List, Dict, Tuple, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# pinecone client (server SDK wrapper)
from pinecone import Pinecone as PineconeClient

# Optional OCR fallback
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Config ----------
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR", "data/laws")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")  # optional
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/LaBSE")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "ingest_artifacts")
CHUNK_TEXT_SAVE_THRESHOLD = int(os.environ.get("CHUNK_TEXT_SAVE_THRESHOLD", 8000))
MIN_CHUNK_LEN = int(os.environ.get("MIN_CHUNK_LEN", 10))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "false").lower() in ("1", "true", "yes")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "chunks"), exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest")

# ---------- Utilities ----------
bn_to_en_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_known_footers_and_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"bdlaws\.minlaw\.gov\.bd\S*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?\s*\d+\s*/\s*\d+\s*\n?", "\n", text)
    text = "\n".join([ln for ln in text.splitlines() if not re.search(r"copyright|legislative|ministry of law|bdlaws", ln, re.I)])
    return normalize_whitespace(text)

def detect_lang(text: str) -> str:
    return "bn" if re.search(r"[\u0980-\u09FF]", text) else "en"

def make_doc_id(filename: str, page: int, chunk_index: int, chunk_preview: str) -> str:
    base = f"{filename}|p{page}|c{chunk_index}|{chunk_preview[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

def safe_translate(text: str, target: str, context: str = "", retries: int = 3) -> str:
    if not text or not text.strip():
        return text
    try:
        if re.fullmatch(r"[\s\d\W/().,-]+", text):
            return text
        attempt = 0
        while attempt < retries:
            try:
                to_translate = (f"{context}: {text}" if context and len(text) > 20 else text)
                res = GoogleTranslator(source="auto", target=target).translate(to_translate)
                if context and res.lower().startswith(context.lower()):
                    res = re.sub(rf"^{re.escape(context)}\s*[:\-–]\s*", "", res, flags=re.I)
                time.sleep(0.12)
                return res
            except Exception as e:
                attempt += 1
                time.sleep(0.25 * attempt)
        log.warning("Translation failed for text start: %s", text[:80])
        return text
    except Exception as e:
        log.exception("Unexpected translation error: %s", e)
        return text

# ---------- PDF loader with optional OCR ----------
def load_pdf_fulltext(path: str):
    doc = fitz.open(path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        try:
            # Some type stubs for PyMuPDF do not include get_text; cast to Any to satisfy type checkers.
            page_any = page  # type: Any
            text = page_any.get_text("text") or ""
            text = strip_known_footers_and_urls(text)
            garbage_ratio = text.count("�") / (len(text) + 1)
            if (len(text.strip()) < 25 or garbage_ratio > 0.03) and OCR_AVAILABLE:
                log.info("Page %d looks poor; attempting OCR for %s", i+1, os.path.basename(path))
                # Prefer new get_pixmap API; fall back to legacy getPixmap for older PyMuPDF versions.
                pix_getter = getattr(page_any, "get_pixmap", None) or getattr(page_any, "getPixmap", None)
                if pix_getter is None:
                    raise RuntimeError("PyMuPDF Page object has no get_pixmap/getPixmap method")
                pix = pix_getter(dpi=200)
                img_path = os.path.join(ARTIFACTS_DIR, f"ocr_page_{os.path.basename(path)}_{i+1}.png")
                pix.save(img_path)
                try:
                    ocr_text = pytesseract.image_to_string(Image.open(img_path), lang="ben+eng")
                    ocr_text = strip_known_footers_and_urls(ocr_text)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                except Exception as e:
                    log.warning("OCR failed for %s page %d: %s", path, i+1, e)
            text = normalize_whitespace(text)
        except Exception as e:
            log.exception("Error extracting page %d from %s: %s", i+1, path, e)
            text = ""
        yield i + 1, text

# ---------- Title & act extraction ----------
ENG_TITLE_PATTERNS = [
    re.compile(r'^(The\s+[\w\-\.,\(\) ]+?(?:Act|Code|Ordinance|Regulation|Rules)[\w \,\:\(\)]*?\d{3,4})', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(.*?)\n\(\s*ACT\s+NO\.', re.IGNORECASE | re.MULTILINE),
]
ENG_ACTNO_PATTERNS = [
    re.compile(r'\(Act\s+No\.?\s*([IVXLC\d]+)\s*of\s*(\d{3,4})\)', re.IGNORECASE),
    re.compile(r'ACT NO\.?\s*([IVXLC\d]+)\s*OF\s*(\d{3,4})', re.IGNORECASE),
]
BN_TITLE_PATTERNS = [ re.compile(r'^(.*?)\n', re.MULTILINE | re.DOTALL) ]
BN_ACTNO_PATTERNS = [ re.compile(r'(\d{3,4}\s*সনের\s*\d+\s*নং)', re.IGNORECASE), re.compile(r'(\d+\s*নং\s*আইন)', re.IGNORECASE) ]

def extract_law_title_and_number(pdf_path: str, first_page_text: str) -> Tuple[str, str]:
    filename = os.path.basename(pdf_path)
    fp = normalize_whitespace(first_page_text or "")
    for p in ENG_TITLE_PATTERNS:
        m = p.search(fp)
        if m:
            title = m.group(1).strip()
            for a in ENG_ACTNO_PATTERNS:
                am = a.search(fp)
                if am:
                    act = f"Act No. {am.group(1)} of {am.group(2)}"
                    return title, act
            return title, ""
    for p in BN_TITLE_PATTERNS:
        m = p.search(fp)
        if m:
            title = m.group(1).strip()
            for a in BN_ACTNO_PATTERNS:
                am = a.search(fp)
                if am:
                    return title, am.group(1).strip()
            return title, ""
    fallback = os.path.splitext(filename)[0].replace('_',' ').replace('-',' ')
    return fallback, ""

# ---------- Splitter ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,
                                          separators=["\n\n", "\n", "।", ".", ";", ":", " "])

# ---------- Embeddings ----------
log.info("Loading embedding model: %s", EMBEDDING_MODEL)
emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------- Pinecone client (NO index creation here) ----------
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env")

if PINECONE_ENV:
    pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
else:
    pc = PineconeClient(api_key=PINECONE_API_KEY)

idx = pc.Index(PINECONE_INDEX_NAME)

# ---------- Build documents ----------
def build_documents(data_dir: str) -> List[Dict]:
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    log.info("Found %d PDF files in %s", len(pdf_files), data_dir)
    docs = []
    report = {}
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        log.info("Processing file: %s", fname)
        try:
            first_page_text = next((t for p,t in load_pdf_fulltext(pdf_path) if p==1), "")
            title_raw, act_raw = extract_law_title_and_number(pdf_path, first_page_text)
            title_lang = detect_lang(title_raw)
            if title_lang == "bn":
                title_bn = title_raw
                title_en = safe_translate(title_raw, "en", context="Act title")
                title_field = title_bn
            else:
                title_en = title_raw
                title_bn = safe_translate(title_raw, "bn", context="আইন")
                title_field = title_en

            act_number_field = ""
            if act_raw:
                if detect_lang(act_raw) == "bn":
                    act_bn = act_raw
                    act_en = safe_translate(act_raw, "en", context="Act number")
                else:
                    act_en = act_raw
                    act_bn = safe_translate(act_raw, "bn", context="আইন নম্বর")
                act_number_field = f"{act_bn} / {act_en}"

            per_file_chunks = 0
            for page_no, page_text in load_pdf_fulltext(pdf_path):
                if not page_text or not page_text.strip():
                    continue

                # extract section/chapter heuristics
                section = ""
                sec_en = re.search(r'Section\s+(\d+)', page_text, re.IGNORECASE)
                sec_bn = re.search(r'ধারা\s+([০-৯]+|\d+)', page_text)
                if sec_bn:
                    sec_bn_en = sec_bn.group(1).translate(bn_to_en_digits)
                    section = f"{sec_bn_en} / ধারা {sec_bn_en}"
                elif sec_en:
                    section = f"{sec_en.group(1)} / Section {sec_en.group(1)}"

                chapter = ""
                chap_en = re.search(r'Chapter\s+([IVXLC]+)', page_text, re.IGNORECASE)
                chap_bn = re.search(r'অধ্যায়\s+([০-৯]+|\d+)', page_text)
                if chap_bn:
                    chapter = f"অধ্যায় {chap_bn.group(1).translate(bn_to_en_digits)}"
                elif chap_en:
                    chapter = f"Chapter {chap_en.group(1)}"

                chunks = splitter.split_text(page_text)
                for cidx, chunk in enumerate(chunks):
                    if not chunk or len(chunk.strip()) < MIN_CHUNK_LEN:
                        if not re.search(r'(ধারা|Section|অধ্যায়|Chapter)', chunk):
                            continue

                    chunk_lang = detect_lang(chunk)
                    lang_field = f"{chunk_lang} / {'en' if chunk_lang=='bn' else 'bn'}"

                    doc_id = make_doc_id(fname, page_no, per_file_chunks, chunk[:200])

                    metadata = {
                        "doc_id": doc_id,
                        "title": title_field,
                        "title-bn": title_bn,
                        "title-en": title_en,
                        "chapter": chapter,
                        "section": section,
                        "act-number": act_number_field,
                        "source": fname,
                        "page": page_no,
                        "lang": lang_field
                    }

                    full_text = chunk.strip()
                    if len(full_text) > CHUNK_TEXT_SAVE_THRESHOLD:
                        chunk_path = os.path.join(ARTIFACTS_DIR, "chunks", f"{doc_id}.json")
                        with open(chunk_path, "w", encoding="utf-8") as fh:
                            json.dump({"doc_id": doc_id, "text": full_text, "metadata": metadata}, fh, ensure_ascii=False)
                        metadata["text_path"] = chunk_path
                        metadata["text"] = full_text[:1000] + " ... [TRUNCATED - full text saved to text_path]"
                    else:
                        metadata["text"] = full_text

                    docs.append({"id": doc_id, "text": full_text, "metadata": metadata})
                    per_file_chunks += 1

            report[fname] = {"chunks": per_file_chunks, "title": title_field[:200], "act": act_number_field[:200]}
            log.info("Prepared %d chunks from %s", per_file_chunks, fname)
        except Exception as e:
            log.exception("Failed to process %s: %s", fname, e)
            report[fname] = {"error": str(e)}

    with open(os.path.join(ARTIFACTS_DIR, "ingest_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    log.info("Total chunks prepared: %d", len(docs))
    return docs

# ---------- Upsert helper ----------
def fetch_existing_ids(id_list: List[str]) -> set:
    """Return set of ids that already exist in the index (calls fetch in batches)."""
    found = set()
    CHUNK = 100
    for i in range(0, len(id_list), CHUNK):
        try:
            resp = idx.fetch(ids=id_list[i:i+CHUNK])
            # FetchResponse exposes 'vectors' as an attribute rather than supporting dict .get()
            vectors = getattr(resp, "vectors", None)
            if vectors:
                found.update(vectors.keys())
        except Exception as e:
            log.warning("fetch existing ids chunk failed: %s", e)
    return found

def upsert_documents_to_pinecone(docs: List[Dict], batch_size: int = BATCH_SIZE):
    if len(docs) == 0:
        log.info("No docs to upsert.")
        return

    # Optional skip check
    if SKIP_EXISTING:
        candidate_ids = [d["id"] for d in docs]
        existing = fetch_existing_ids(candidate_ids)
        if existing:
            log.info("Skipping %d already-existing docs (SKIP_EXISTING enabled).", len(existing))
            docs = [d for d in docs if d["id"] not in existing]

    if not docs:
        log.info("No new docs to upsert after skip check.")
        return

    # test embed for dimension
    try:
        test_vec = emb.embed_documents([docs[0]["text"]])[0]
    except Exception as e:
        log.exception("Embedding failed on sample text: %s", e)
        raise
    vec_dim = len(test_vec)
    log.info("Embedding dim: %d", vec_dim)

    audit_path = os.path.join(ARTIFACTS_DIR, "upsert_audit.jsonl")
    fh_audit = open(audit_path, "a", encoding="utf-8")

    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i:i+batch_size]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        try:
            vectors = emb.embed_documents(texts)
            payload = []
            for d, v in zip(batch, vectors):
                v_list = list(map(float, v))
                md = d["metadata"]
                payload.append({"id": d["id"], "values": v_list, "metadata": md})
                audit_entry = {"id": d["id"], "meta": {"source": md.get("source"), "page": md.get("page"), "doc_id": md.get("doc_id")}, "vector_head": v_list[:8]}
                fh_audit.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

            idx.upsert(vectors=payload)
            log.info("Upserted %d/%d", min(i + batch_size, total), total)
        except Exception as e:
            log.exception("Failed embedding/upsert for batch %d-%d: %s", i, i+batch_size-1, e)
        time.sleep(0.15)

    fh_audit.close()
    log.info("Upsert audit saved to %s", audit_path)

# ---------- Main ----------
if __name__ == "__main__":
    documents = build_documents(DATA_DIR)
    upsert_documents_to_pinecone(documents)
    log.info("Done.")
