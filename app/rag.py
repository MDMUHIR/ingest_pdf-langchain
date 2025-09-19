import os, re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# HuggingFace embeddings (must match ingestion)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Pinecone retriever
index_name = os.environ["PINECONE_INDEX_NAME"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Gemini for answering
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Prompt template
SYSTEM_PROMPT = """You are a legal information assistant for Bangladeshi law.
Rules:
- Do NOT give personal legal advice, only provide information from the laws.
- Always cite sources like (Act title, page).
- If unsure, say "I don't have enough information".
- Answer in the same language as the question (Bangla if Bangla, English if English)."""

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

rag_chain = (
    {"question": RunnablePassthrough(),
     "context": retriever | format_docs}
    | prompt
    | llm
    | StrOutputParser()
)

# test
# if __name__ == "__main__":
#     # Test query (English)
#     q1 = "What does Section 420 of Penal Code 1860 state?"
#     print("\n🔎 Query:", q1)
#     print("💡 Answer:\n", rag_chain.invoke(q1))

#     # Test query (Bangla)
#     q2 = "দণ্ডবিধি ১৮৬০-এর ধারা ৪২০ সম্পর্কে তথ্য দিন।"
#     print("\n🔎 Query:", q2)
#     print("💡 Answer:\n", rag_chain.invoke(q2))

if __name__ == "__main__":
    # Debug retrieval for English
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("Section 420 of Penal Code 1860")
    print("\n📌 Retrieved Chunks for English Query:")
    for d in docs:
        print("----")
        print("Title:", d.metadata.get("title"))
        print("Page:", d.metadata.get("page"))
        print("Lang:", d.metadata.get("lang"))
        print("Preview:", d.page_content[:250], "...\n")

    # Debug retrieval for Bangla
    docs_bn = retriever.invoke("ধারা ৪২০ দণ্ডবিধি ১৮৬০")
    print("\n📌 Retrieved Chunks for Bangla Query:")
    for d in docs_bn:
        print("----")
        print("Title:", d.metadata.get("title"))
        print("Page:", d.metadata.get("page"))
        print("Lang:", d.metadata.get("lang"))
        print("Preview:", d.page_content[:250], "...\n")

