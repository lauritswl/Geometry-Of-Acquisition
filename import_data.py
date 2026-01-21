
# %%
import pandas as pd
print("Pandas version:", pd.__version__)
# %%
# Load datasets from HuggingFace (only learner corpora for now)
df_elle_et = pd.read_json("hf://datasets/UniversalCEFR/elle_et/elle.json")
df_icle500_en = pd.read_json("hf://datasets/UniversalCEFR/icle500_en/icle500.json")
df_merlin_cs = pd.read_json("hf://datasets/UniversalCEFR/merlin_cs/merlin-cs.json")
df_merlin_it = pd.read_json("hf://datasets/UniversalCEFR/merlin_it/merlin-it.json")
df_merling_de = pd.read_json("hf://datasets/UniversalCEFR/merlin_de/merlin-de.json")
df_cefr_asag_en = pd.read_json("hf://datasets/UniversalCEFR/cefr_asag_en/cefr_asag.json")
df_cople2_pt = pd.read_json("hf://datasets/UniversalCEFR/cople2_pt/cople2.json")
# %% 
# Cleaning functions
import re
def strip_title(text: str) -> str:
    # Remove a leading line starting with "Title:" (case-insensitive), plus one blank line after it if present
    return re.sub(r"(?im)^title:.*\n\s*\n?", "", text, count=1).strip()

# %%
## Manual: Inspect datasets and clean as needed
# Remove all df_elle_et rows where text starts with "I OSA."
df_elle_et = df_elle_et[~df_elle_et["text"].str.startswith("I OSA.")]
df_icle500_en["text"] = df_icle500_en["text"].apply(strip_title)

# %%
# Combine all datasets into a single DataFrame
df_all = pd.concat([
    df_elle_et,
    df_icle500_en,
    df_merlin_cs,
    df_merlin_it,
    df_merling_de,
    df_cefr_asag_en,
    df_cople2_pt
], ignore_index=True)


# Show list of all unique CEFR levels in the combined DataFrame with counts
print("Unique CEFR levels in combined dataset:")
print(df_all["cefr_level"].value_counts().sort_index())

# Remove Empty, NA and unrated entries
df_all = df_all[~df_all["cefr_level"].isin(["NA", "unrated", "EMPTY"])]
# Convert B1+ and B2+ into B1 and B2 respectively
df_all["cefr_level"] = df_all["cefr_level"].replace({"B1+": "B1", "B2+": "B2"})
# Store cefr_level as an ordered categorical variable
cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
df_all["cefr_level"] = pd.Categorical(df_all["cefr_level"], categories=cefr_order, ordered=True)

# Show updated list of unique CEFR levels in the cleaned combined DataFrame with counts
print("Unique CEFR levels in cleaned combined dataset:")
print(df_all["cefr_level"].value_counts().sort_index())

# %%
# Show count of each cefr_level for each language
print("CEFR level distribution by language:")
print(df_all.groupby("lang")["cefr_level"].value_counts().unstack().fillna(0))


# %%
# Create a test/train split (80/20) stratified by cefr_level
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["cefr_level"], random_state=89)
print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")


# %%
# Save cleaned combined DataFrame to a CSV file
train_df.to_csv("/work/Code/Geometry-Of-Acquisition/data/train_cefr_dataset.csv", index=False)
test_df.to_csv("/work/Code/Geometry-Of-Acquisition/data/test_cefr_dataset.csv", index=False)

# %%
