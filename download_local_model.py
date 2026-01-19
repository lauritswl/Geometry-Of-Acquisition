from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login

MODEL_ID = "google/embeddinggemma-300m"
LOCAL_DIR = "./embeddinggemma-300m"

# Optional: only needed if you didn't run `huggingface-cli login`
# login(token="hf_your_token_here")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=True,
)

# Load model
model = AutoModel.from_pretrained(
    MODEL_ID,
    use_auth_token=True,
)

# Save locally
tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)

print(f"Model saved to {LOCAL_DIR}")
