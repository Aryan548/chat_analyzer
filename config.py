import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is loaded correctly
if GEMINI_API_KEY:
    print("✅ Gemini API key loaded successfully!")
else:
    print("❌ Error: Gemini API key not found. Please check your .env file.")
