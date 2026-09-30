import chromadb


# ==========================================
# 1. CONNECT TO PERSISTENT CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ==========================================
# 2. CREATE / OPEN COLLECTION
# ==========================================

collection = client.get_or_create_collection(
    name="policies",
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 3. KNOWLEDGE BASE
# ==========================================

documents = [
    "Refunds are processed within five business days.",

    "You can return any item within 30 days of delivery.",

    "Our support team replies to tickets within four hours.",

    "Enterprise plans include a dedicated account manager.",

    "The battery lasts up to 12 hours on a single charge.",

    "Travel expense reimbursement reports must be submitted within 14 business days of trip completion."
]


metadatas = [
    {"topic": "billing"},
    {"topic": "billing"},
    {"topic": "support"},
    {"topic": "support"},
    {"topic": "hardware"},
    {"topic": "finance"},
]


ids = [
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]


# ==========================================
# 4. INSERT DOCUMENTS
# ==========================================

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)


print("ChromaDB seeded successfully.")
print("Documents:", collection.count())