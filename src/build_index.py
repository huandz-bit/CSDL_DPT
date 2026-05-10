import faiss
import numpy as np

from src.mongodb_store import collection

docs = list(collection.find())

embeddings = []

ids = []

for doc in docs:

    emb = doc["features"]["speaker_embedding"]

    embeddings.append(emb)

    ids.append(str(doc["_id"]))

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "indexes/voice.index"
)

np.save(
    "indexes/ids.npy",
    ids
)

print("FAISS index built.")