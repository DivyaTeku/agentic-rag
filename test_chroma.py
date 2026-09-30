import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

kb = client.get_collection("policies")


query = "How long does it take to get my money back?"

results = kb.query(
    query_texts=[query],
    n_results=2
)


print("\nResults:\n")

for document, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Score: {1 - distance:.3f}")
    print(document)
    print()