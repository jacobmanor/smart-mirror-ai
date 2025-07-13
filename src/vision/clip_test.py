from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define outfit labels to compare
labels = ["a person in a red hoodie", "a person in business casual", 
          "someone in gym clothes", "formalwear", "jeans and a white shirt"]

# Load image
image = Image.open("data/fit1.jpg")

# Run through CLIP
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

# Print match
best_idx = torch.argmax(probs)
print(f"Most likely: {labels[best_idx]} ({probs[0][best_idx].item()*100:.1f}%)")
