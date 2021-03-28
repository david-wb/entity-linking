import torch
from scipy.spatial.distance import cosine

from transformers import AutoModel, AutoTokenizer

# Load the model
tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
model = AutoModel.from_pretrained("johngiorgi/declutr-base")

# Prepare some text to embed
text = [
    "A smiling costumed woman is holding an umbrella.",
    "A happy woman in a fairy costume holds an umbrella.",
]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Embed the text
with torch.no_grad():
    sequence_output = model(**inputs)[0]

# Mean pool the token-level embeddings to get sentence-level embeddings
embeddings = torch.sum(
    sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

# Compute a semantic similarity via the cosine distance
semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
