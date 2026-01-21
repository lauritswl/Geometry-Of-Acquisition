
# %% Import Modules
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
print("Modules imported successfully.")

# %%
# Define save paths
train_path = "/work/Code/Geometry-Of-Acquisition/data/embeddings/train_embeddings.csv"
test_path = "/work/Code/Geometry-Of-Acquisition/data/embeddings/test_embeddings.csv"

import os
# Make sure the directories exist
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)


# %% Load Data
# Load train and test data from f"data/{split}_cefr_dataset.csv"
data = {}
for split in ["train", "test"]:
    data[split] = pd.read_csv(f"/work/Code/Geometry-Of-Acquisition/data/{split}_cefr_dataset.csv")
print("Data loaded successfully.")

# %%
# Load model
model = SentenceTransformer("google/embeddinggemma-300m")

# Convert train/test texts to lists
train_documents = data["train"]["text"].tolist()
test_documents = data["test"]["text"].tolist()

# Encode documents
train_embeddings = model.encode(train_documents, show_progress_bar=True)
test_embeddings = model.encode(test_documents, show_progress_bar=True)

# Convert embeddings to DataFrame
train_df = pd.DataFrame(train_embeddings)
test_df = pd.DataFrame(test_embeddings)

# Save embeddings to CSV
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)


print("Embeddings saved successfully")
