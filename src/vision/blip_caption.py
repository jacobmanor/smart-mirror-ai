from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Example usage
if __name__ == "__main__":
    test_image = os.path.join(os.path.dirname(__file__), "../../data/fit1.jpg")
    print("🧥 Generated Caption:", generate_caption(test_image))
