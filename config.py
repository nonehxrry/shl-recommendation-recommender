# config.py
import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if openai_key:
    print("✅ OpenAI key loaded:", openai_key[:10] + "...")
else:
    print("❌ OpenAI key not found.")

if gemini_key:
    print("✅ Gemini key loaded:", gemini_key[:10] + "...")
else:
    print("❌ Gemini key not found.")
