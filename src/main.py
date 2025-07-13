from vision.blip_caption import generate_caption
from assistant.gpt_wrapper import get_style_feedback
import os

if __name__ == "__main__":
    # Replace this with any test image
    image_file = "data/fit3.jpg"
    
    print("📸 Scanning outfit...")
    caption = generate_caption(image_file)
    print("🧥 BLIP says:", caption)

    # Simulate user question
    user_input = "Thoughts on this for a job interview?"

    print("\n🤖 Thinking...")
    gpt_response = get_style_feedback(caption, user_input)

    print("\n🪞 MirrorMate says:", gpt_response)
