from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

# Load model and processor
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Define possible outfit labels
labels = [
    "a person in a light blue dress shirt and brown khakis",
    "a person in a purple shirt and jeans",
    "a person in business casual clothes",
    "someone wearing gym clothes",
    "a person in a white shirt and khakis",
    "formalwear",
    "summer outfit with shorts",
    "winter coat and boots",
]

# Loop through your data/ folder images
image_dir = os.path.join(os.path.dirname(__file__), "../../data")
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

for file in image_files:
    print(f"\n🔍 Processing {file}")
    image = Image.open(os.path.join(image_dir, file))

    # Preprocess for CLIP
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    # Get top result
    best_idx = torch.argmax(probs)
    print(f"🧠 Best match: {labels[best_idx]} ({probs[0][best_idx].item()*100:.1f}%)")
