import google.generativeai as genai
import os
from dotenv import load_dotenv

# ✅ Load API key securely from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Correct Gemini model name
GEMINI_MODEL = "gemini-1.5-pro"

def summarize_chat(df, max_words=250):
    """Summarizes a given WhatsApp chat using Gemini API (limits summary to 250 words)."""
    
    if df.empty or "message" not in df.columns:
        return "No chat data available for summarization."
    
    # ✅ Remove "<Media omitted>" messages
    chat_text = " ".join(df["message"].dropna())
    chat_text = chat_text.replace("<Media omitted>", "")

    # ✅ Limit input size for API efficiency
    chat_text = chat_text[:6000]  

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(f"Summarize the following chat in {max_words} words:\n\n{chat_text}")
        
        # ✅ Extract and trim summary if it exceeds the word limit
        summary = response.text.strip()
        if len(summary.split()) > max_words:
            summary = " ".join(summary.split()[:max_words])  

        return summary

    except Exception as e:
        return f"❌ Error generating summary: {e}"
