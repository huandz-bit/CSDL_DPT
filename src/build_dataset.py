import os

from src.preprocess import preprocess_audio
from src.feature_extractor import extract_features
from src.mongodb_store import collection

DATASET_DIR = "dataset"
PROCESSED_DIR = "processed"

for file in os.listdir(DATASET_DIR):

    input_path = os.path.join(DATASET_DIR, file)

    output_path = os.path.join(PROCESSED_DIR, file)

    preprocess_audio(
        input_path,
        output_path
    )

    features = extract_features(output_path)

    doc = {

        "filename": file,

        "path": output_path,

        "features": features
    }

    collection.insert_one(doc)

    print("Inserted:", file)