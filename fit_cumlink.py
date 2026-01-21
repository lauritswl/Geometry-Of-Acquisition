# %% Import Modules
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.vector import ConceptVector
print("Modules imported successfully.")

# %% Load Data
# Load train and test data from f"data/{split}_cefr_dataset.csv"
data = {}
for split in ["train", "test"]:
    data[split] = pd.read_csv(f"/work/Code/Geometry-Of-Acquisition/data/{split}_cefr_dataset.csv")
print("Data loaded successfully.")

# %%
# Define paths
embeddings = {}
for split in ["train", "test"]:
    embeddings[split] = pd.read_csv(f"/work/Code/Geometry-Of-Acquisition/data/embeddings/{split}_embeddings.csv")

print("Embeddings loaded successfully:")
print(f"Train embeddings shape: {embeddings["train"].shape}")
print(f"Test embeddings shape: {embeddings["test"].shape}")
# %%

# %% Fit ordinal regression model using mord
X = embeddings["train"]
y = data["train"]["cefr_level"].values
# order y categories as A1 < A2 < B1 < B2 < C1 < C2
level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

# Fit a mord.LogisticIT model
import mord
model = mord.LogisticIT()
model.fit(X.to_numpy(), np.array([level_order[level] for level in y]))


# %% Evaluate model on test set
y_pred = model.predict(embeddings["test"].to_numpy())
# transform data["test"["cefr_level"] to numerical labels
y_test = np.array([level_order[level] for level in data["test"]["cefr_level"].values])
# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Test accuracy:", accuracy)

# %%
# Coefficients for predictors
print("Number of coefficients:", len(model.coef_))

# Cutoff points between classes
print("Cutoff points between classes:", model.theta_)

#%% 
# Initialize ConceptVector
cv = ConceptVector(normalize=False)

# Directly set the vector to your model coefficients
cv.vector = model.coef_
cv.original_vector = model.coef_
cv.dim = model.coef_.shape[0]

# Optionally normalize
if cv.normalize:
    cv.normalize_vector()

# Now you can project new embeddings
new_embeddings = embeddings["test"].to_numpy()
projections = cv.project(new_embeddings)

print("Projection shape:", projections.shape)
print("First 5 projections:", projections[:5])


# %% Save distribution of projections by class with thresholds
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Make sure the output folder exists
os.makedirs("plots", exist_ok=True)

# Create a DataFrame with CEFR levels and projection values
df_proj = pd.DataFrame({
    "cefr_level": data["test"]["cefr_level"],
    "projection": projections
})

# Optional: order the classes
cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Get thresholds from ordinal model
theta = model.theta_  # shape: (n_classes - 1,)
theta_labels = cefr_order[1:]  # associate each threshold with the next class

# --- Violin plot ---
plt.figure(figsize=(10, 6))
sns.violinplot(x="cefr_level", y="projection", data=df_proj, order=cefr_order, inner="quartile")
plt.title("Distribution of Model Coefficient Projections by CEFR Class")
plt.xlabel("CEFR Level")
plt.ylabel("Projection onto Coefficients")

# Add threshold lines
for t, label in zip(theta, theta_labels):
    plt.axhline(y=t, color='red', linestyle='--', alpha=0.7)
    plt.text(x=len(cefr_order)-0.5, y=t, s=f'{label}', color='red', va='center', ha='left')

plt.tight_layout()
plt.savefig("/work/Code/Geometry-Of-Acquisition/plots/projections_violin_with_thresholds.png")
plt.close()

# --- Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="cefr_level", y="projection", data=df_proj, order=cefr_order)
plt.title("Distribution of Model Coefficient Projections by CEFR Class (Boxplot)")
plt.xlabel("CEFR Level")
plt.ylabel("Projection onto Coefficients")

# Add threshold lines
for t, label in zip(theta, theta_labels):
    plt.axhline(y=t, color='red', linestyle='--', alpha=0.7)
    plt.text(x=len(cefr_order)-0.5, y=t, s=f'{label}', color='red', va='center', ha='left')

plt.tight_layout()
plt.savefig("/work/Code/Geometry-Of-Acquisition/plots/projections_boxplot_with_thresholds.png")
plt.close()


# %%
# %% Confusion matrix with totals
import pandas as pd
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(level_order))))

# Convert to DataFrame for readability
class_names = ["A1", "A2", "B1", "B2", "C1", "C2"]
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Add a column for row totals (true class counts)
cm_df["Total (true)"] = cm_df.sum(axis=1)

# Add a row for column totals (predicted counts)
totals = cm_df.sum(axis=0)
totals.name = "Total (pred)"
cm_df = pd.concat([cm_df, totals.to_frame().T])

print("Confusion Matrix with Totals:")
print(cm_df)




# %% F1 scores
from sklearn.metrics import f1_score, classification_report

# Class names in order
class_names = ["A1", "A2", "B1", "B2", "C1", "C2"]

# F1 score per class
f1_per_class = f1_score(y_test, y_pred, labels=list(range(len(class_names))), average=None)
for cls, score in zip(class_names, f1_per_class):
    print(f"F1 score for {cls}: {score:.3f}")

# Macro and weighted F1
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
print(f"\nMacro F1: {f1_macro:.3f}")
print(f"Weighted F1: {f1_weighted:.3f}")

# Optional: full classification report
print("\nFull Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
