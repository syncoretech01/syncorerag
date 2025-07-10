import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
import re


# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # Match your app.py
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "syncoretech-index")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "syncore-rag")

# === Initialize clients ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists (matching app.py configuration)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX)

# === Text preprocessing for better relevance ===
def preprocess_text(text):
    """Clean and normalize text for better embedding quality"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation for context
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    # Trim
    text = text.strip()
    return text

# === Load chunks from file ===
def load_chunks(filename="chunks.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# === Generate embeddings in batches ===
def embed_texts(chunks):
    print(f"[+] Embedding {len(chunks)} chunks...")
    texts = [preprocess_text(chunk["text"]) for chunk in chunks]
    embeddings = []
    batch_size = 100

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"  # CRITICAL: Match app.py!
            )
            batch_embeddings = [item.embedding for item in response.data]
        except Exception as e:
            print(f"[-] Failed to embed batch {i}: {e}")
            # Skip failed batches instead of using zero vectors
            continue

        embeddings.extend(batch_embeddings)

    return embeddings

# === Enhanced metadata for better context ===
def create_enhanced_metadata(chunk):
    """Create rich metadata for better retrieval"""
    text = chunk["text"]
    metadata = {
        "text": text[:2000],  # Pinecone metadata size limit
        "source": chunk.get("source", "unknown"),
        "chunk_id": chunk.get("id", str(uuid4())),
        "word_count": len(text.split()),
        "char_count": len(text)
    }
    
    # Add content type hints for better filtering
    if any(keyword in text.lower() for keyword in ["price", "cost", "$", "pricing", "fee"]):
        metadata["content_type"] = "pricing"
    elif any(keyword in text.lower() for keyword in ["contact", "email", "phone", "address"]):
        metadata["content_type"] = "contact"
    elif any(keyword in text.lower() for keyword in ["about", "mission", "vision", "team"]):
        metadata["content_type"] = "about"
    elif any(keyword in text.lower() for keyword in ["service", "product", "offer", "solution"]):
        metadata["content_type"] = "services"
    else:
        metadata["content_type"] = "general"
    
    return metadata

# === Upload embeddings to Pinecone ===
def upsert_to_pinecone(chunks, embeddings, namespace):
    print(f"[+] Uploading vectors to Pinecone namespace '{namespace}'...")

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        if not embedding or len(embedding) != 1536:
            continue
        
        vectors.append({
            "id": chunk.get("id", str(uuid4())),
            "values": embedding,
            "metadata": create_enhanced_metadata(chunk)
        })

    # Upload in batches to avoid timeouts
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i:i + batch_size]
        if batch:
            index.upsert(vectors=batch, namespace=namespace)
    
    print(f"[✓] Uploaded {len(vectors)} vectors to Pinecone.")

# === Main flow ===
if __name__ == "__main__":
    # Clear existing vectors in namespace (optional)
    print(f"[!] Clearing existing vectors in namespace '{NAMESPACE}'...")
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except:
        pass
    
    chunks = load_chunks()
    if not chunks:
        print("[-] No chunks found.")
        exit(1)

    embeddings = embed_texts(chunks)
    if len(embeddings) != len(chunks):
        print(f"[!] Warning: Only embedded {len(embeddings)} out of {len(chunks)} chunks")
        chunks = chunks[:len(embeddings)]  # Match chunks to embeddings
    
    upsert_to_pinecone(chunks, embeddings, NAMESPACE)