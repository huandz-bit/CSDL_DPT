import faiss
import numpy as np

from src.feature_extractor import extract_features
from src.mongodb_store import collection
from bson import ObjectId

index = faiss.read_index(
    "indexes/voice.index"
)

ids = np.load(
    "indexes/ids.npy",
    allow_pickle=True
)

def search_voice(audio_path):

    features = extract_features(audio_path)

    query = np.array(
        [features["speaker_embedding"]],
        dtype=np.float32
    )

    D, I = index.search(query, k=5)

    results = []

    for score, idx in zip(D[0], I[0]):

        mongo_id = ids[idx]

        doc = collection.find_one({
    "_id": ObjectId(mongo_id)
})

        results.append({

            "filename": doc["filename"],

            "similarity": float(score)
        })

    return results