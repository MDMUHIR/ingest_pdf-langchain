# scripts/ingest_pdfs.py
import os, glob, hashlib, re, time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
DATA_DIR = "data/laws"

# ---------------- PDF Loader ----------------
def load_pdf_fulltext(path):
    """Extracts page-wise text from PDF without losing Unicode chars."""
    doc = fitz.open(path)
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        # Normalize whitespace but keep section breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)  # preserve paragraph breaks
        yield page_no + 1, text.strip()

# ---------------- Text Splitter ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", "।", ".", ";", ":", " "],
)

# ---------------- Embeddings (FREE) ----------------
# multilingual model (Bangla + English), 384-dim vectors
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ---------------- Vector Store ----------------
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb)

# ---------------- Helpers ----------------
def detect_lang(text: str) -> str:
    """Detect Bangla or English chunk."""
    return "bn" if re.search(r"[\u0980-\u09FF]", text) else "en"

def make_doc_id(content, meta):
    h = hashlib.sha256((content + str(meta)).encode("utf-8")).hexdigest()
    return h[:32]

# ---------------- Prepare Documents ----------------
all_docs = []
for pdf in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
    title = os.path.splitext(os.path.basename(pdf))[0]
    for page, text in load_pdf_fulltext(pdf):
        if not text or len(text.strip()) < 15:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source_path": pdf,
                "title": title,
                "page": page,
                "chunk": i,
                "lang": detect_lang(chunk),
            }
            all_docs.append(Document(page_content=chunk, metadata=metadata))

print(f"Prepared {len(all_docs)} chunks from {len(glob.glob(os.path.join(DATA_DIR, '*.pdf')))} files.")

# ---------------- Upload to Pinecone ----------------
BATCH = 64  # safe batch size
for i in range(0, len(all_docs), BATCH):
    batch = all_docs[i:i+BATCH]
    vectorstore.add_documents(batch)
    print(f"Upserted {i+len(batch)}/{len(all_docs)}")
    time.sleep(1)  # gentle pacing
