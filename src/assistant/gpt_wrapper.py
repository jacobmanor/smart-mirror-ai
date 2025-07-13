from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_style_feedback(outfit_description, user_question):
    prompt = f"""
You are MirrorMate, a stylish, witty, and helpful fashion assistant.

The user is wearing: {outfit_description}
They asked: "{user_question}"

Give short, thoughtful, charming style advice.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    return response["choices"][0]["message"]["content"]