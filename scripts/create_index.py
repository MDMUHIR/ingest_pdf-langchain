# scripts/create_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "bd-laws")
# create only if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        # dimension=384,         
        dimension=768,         
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created:", INDEX_NAME)
else:
    print("Index already exists:", INDEX_NAME)
# scripts/create_index.py
# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# load_dotenv()
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "bd-laws")
# DELETE_EXISTING = os.environ.get("DELETE_EXISTING", "false").lower() == "true"

# # Delete existing index if flag is set
# if DELETE_EXISTING and INDEX_NAME in [i.name for i in pc.list_indexes()]:
#     print(f"Deleting existing index: {INDEX_NAME}...")
#     pc.delete_index(INDEX_NAME)
#     print("Deleted.")

# # Create index if it does not exist
# if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # or 384 if you use multilingual embeddings
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     print("Index created:", INDEX_NAME)
# else:
#     print("Index already exists:", INDEX_NAME)
