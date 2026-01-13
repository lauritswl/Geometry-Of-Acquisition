# %%
# Flags:
# Chose to fit model or load local model
load_local = 1

# %%
import torch
import pandas as pd
from src.embedder import Embedder
from src.vector import ConceptVector
print("Modules imported successfully.")


# %%
# Load local embedding model:
from transformers import AutoTokenizer, AutoModel
model_path = "models/embeddinggemma-300m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print("Model loaded successfully.")

# %%
# Load train and test data
data = {}
for split in ["train", "test"]:
    df = pd.read_csv(f"./data/cefr-sp/wiki-auto/CEFR-SP_wikiauto_{split}.txt", sep="\t", names=["text", "rater_1", "rater_2"])
    df["label"] = df[["rater_1", "rater_2"]].mean(axis=1)
    data[split] = df[["text", "label"]]

print("Data loaded successfully.")



# %%
test=data["train"]["text"].tolist()
inputs = tokenizer(test, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
print("Embedding generated successfully:", embeddings.shape)


# %%
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels as sm
# reset indicies from y
y = data["train"]["label"]
X = embeddings.numpy()

# Change y to pd.Series with ordered CategoricalD type
y = pd.Series(y).astype(pd.CategoricalDtype(categories=sorted(y.unique()), ordered=True))
X = pd.DataFrame(X)

if load_local:
    import pickle
    # load model
    result = pickle.load(open("./models/regressions/cefr_sp_ord_model.pkl", "rb"))
else:
    # Fit ordinal regression model
    Ord_model = OrderedModel(endog=y, exog=X, distr='probit')
    result = Ord_model.fit(method='bfgs')
    # save results
    result.save("./models/regressions/cefr_sp_ord_model.pkl")


print(result.summary())

# %%
# Show predicted table as heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
Dataframe = result.pred_table().iloc[:-1, :-1]
# normalize by row
Dataframe = Dataframe.div(Dataframe.sum(axis=0), axis=1)

# Add labels to x and y axis
categories = y.cat.categories
Dataframe.index = categories
Dataframe.columns = categories
plt.figure(figsize=(8,6))
sns.heatmap(Dataframe, annot=True, cmap='Blues')
plt.title("Predicted vs Actual Labels")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# %%
# Count how many of each unique label in y
print("Label distribution in training set:")
print(y.value_counts().sort_index())


# %%
# Test model on test set
test_y = data["test"]["label"]
test_X_texts = data["test"]["text"].tolist()
test_inputs = tokenizer(test_X_texts, return_tensors="pt", truncation=True,
                        padding=True)
with torch.no_grad():
    test_outputs = model(**test_inputs)
test_embeddings = test_outputs.last_hidden_state.mean(dim=1).squeeze()
test_X = pd.DataFrame(test_embeddings.numpy())  
predicted = result.model.predict(result.params, exog=test_X)
#%%
# Count how many of each unique label in y
print("Label distribution in test set:")
print(test_y.value_counts().sort_index())


# %%
# Convert result.params[:768] to numpy array
params_array = result.params[:768].to_numpy()
prof_CV = ConceptVector()
prof_CV.original_vector = params_array
prof_CV.vector = params_array.copy()
prof_CV.dim = len(params_array)
# Project test embeddings onto concept vector
prof_CV.normalize_vector()
projections = prof_CV.project(test_embeddings.numpy())




# %%
 Show kernel density estimate of projections colored by label
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Map numeric labels to CEFR levels
cefr_mapping = {1.0: 'A1', 1.5: 'A1+', 2.0: 'A2', 2.5: 'A2+', 3.0: 'B1', 3.5: 'B1+', 4.0: 'B2', 4.5: 'B2+', 5.0: 'C1', 5.5: 'C1+', 6.0: 'C2'}
test_y_mapped = test_y.map(cefr_mapping)

df_proj = pd.DataFrame({"projection": projections, "label": test_y_mapped})
plt.figure(figsize=(10,6))
sns.kdeplot(data=df_proj, x="projection", hue="label", fill=True,
            common_norm=False, alpha=0.1)
plt.title("Kernel Density Estimate of Projections by Label")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Density")
plt.show()


#%%
# Show  scatter plot of projections vs labels
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_proj, x="projection", y="label", hue="label", palette="viridis", alpha=0.7)
plt.title("Scatter Plot of Projections vs Labels")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Label")
plt.show()

# %%
# Make a stacked sns histogram, ordered by label with first label at bottom
plt.figure(figsize=(10,6))
sns.histplot(data=df_proj, x="projection", hue="label", palette="husl", multiple="stack", stat="density", common_norm=False, alpha=0.7, bins=100)
plt.title("Stacked Histogram of Projections by Label")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Density")
plt.show()  





# Make a dataframe with text, true label, and projection
df_results = pd.DataFrame({
    "text": test_X_texts,
    "true_label": test_y,
    "projection": projections
})





# %%
# Rerun projection and plot on out of sample sentences: data/CEFR-SP/SCoRE/CEFR-SP_SCoRE_test.txt
from src.vector import ConceptVector
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# %%
# Read data
df_oos = pd.read_csv("./data/kaggle/ielts_writing_dataset.csv")


# %%
oos_texts = df_oos["Essay"].tolist()
oos_labels = df_oos["Overall"]

# Embed using model:

inputs = tokenizer(oos_texts, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
ood_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
print("Embedding generated successfully:", ood_embeddings.shape)

# Project onto concept vector
projections_oos = prof_CV.project(ood_embeddings.numpy())
# Plot KDE of projections colored by label
df_proj_oos = pd.DataFrame({"projection": projections_oos, "label": oos_labels})
plt.figure(figsize=(10,6))
sns.kdeplot(data=df_proj_oos, x="projection", hue="label", fill=True,
            common_norm=False, alpha=0.3)
plt.title("KDE of Projections on Out-of-Sample Data by Label")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Density")
plt.show()

# Plot histogram grouped by label (stacked)
plt.figure(figsize=(10,6))
sns.histplot(data=df_proj_oos, x="projection", hue="label", palette="husl", element="step", stat="density", common_norm=False, alpha=0.3)
plt.title("Histogram of Projections on Out-of-Sample Data by Label")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Density")
plt.show()


# %%
# Make a stacked sns histogram, ordered by label with first label at bottom
plt.figure(figsize=(10,6))
sns.histplot(data=df_proj_oos, x="projection", hue="label", palette="husl", multiple="stack", stat="density", common_norm=False, alpha=0.7)
plt.title("Stacked Histogram of Projections on Out-of-Sample Data by Label")
plt.xlabel("Projection onto Proficiency Concept Vector")
plt.ylabel("Density")
# %%
