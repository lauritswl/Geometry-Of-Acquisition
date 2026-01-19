# %% Import Modules
import torch
import pandas as pd
import numpy as np
import torch
from src.vector import ConceptVector
print("Modules imported successfully.")


# %% Load Model
# Load local embedding model:
from transformers import AutoTokenizer, AutoModel
model_path = "models/embeddinggemma-300m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print("EmbeddingGemma_300m loaded successfully.")

# %% Load Data
# Load train and test data from f"data/{split}_cefr_dataset.csv"
data = {}
for split in ["train", "test"]:
    data[split] = pd.read_csv(f"data/{split}_cefr_dataset.csv")
print("Data loaded successfully.")



# %% Generate embeddings for train and test data

inputs = tokenizer(data["train"]["text"].tolist(), return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
train_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
print("Training Embedding generated successfully:", train_embeddings.shape)
# Save train embeddings
torch.save(train_embeddings, "data/embeddings/train_gemma300m.pt")

inputs = tokenizer(data["test"]["text"].tolist(), return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
test_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
print("Embedding generated successfully:", test_embeddings.shape)
# Save test embeddings
torch.save(test_embeddings, "data/embeddings/test_gemma300m.pt")

# %% Fit ordinal regression model using mord

X = train_embeddings
y = data["train"]["cefr_level"].values
# order y categories as A1 < A2 < B1 < B2 < C1 < C2
level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

# Fit a mord.LogisticIT model
import mord
model = mord.LogisticIT()
model.fit(X.numpy(), np.array([level_order[level] for level in y]))


# %% Evaluate model on test set
y_pred = model.predict(test_embeddings.numpy())
# transform data["test"["cefr_level"] to numerical labels
y_test = np.array([level_order[level] for level in data["test"]["cefr_level"].values])
# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Test accuracy:", accuracy)




# Define vector in object:
prof_CV = ConceptVector()
prof_CV.original_vector = params_array
prof_CV.vector = params_array.copy()
prof_CV.dim = len(params_array)
# Project test embeddings onto concept vector
prof_CV.normalize_vector()
projections = prof_CV.project(test_embeddings.numpy())


# %%
model(torch.as_tensor(test_embeddings))






# %%
