import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd

# ✅ Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def detect_fake_messages(df):
    """Detects fake messages in WhatsApp chat using Google Gemini API and returns flagged messages with sender info."""
    
    if df.empty or "message" not in df.columns or "user" not in df.columns:
        return "No chat data available for analysis.", [], 0.0

    # Extract messages and their senders
    messages = df[["user", "message"]].dropna()

    # ✅ Remove "<Media omitted>" messages
    filtered_messages = messages[messages["message"].str.contains("<Media omitted>") == False]

    if filtered_messages.empty:
        return "No valid messages to analyze.", [], 0.0

    # Convert messages to a structured format for AI processing
    chat_text = "\n".join(f"{row['user']}: {row['message']}" for _, row in filtered_messages.iterrows())

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Use an optimized model for text analysis
        response = model.generate_content(
            f"Analyze the following WhatsApp messages and identify fake information. "
            "For each fake message, return the sender's name and message content. "
            "Additionally, provide the percentage of fake messages in the chat.\n\n"
            f"{chat_text}"
        )

        fake_message_results = response.text.strip()

        # ✅ If no fake messages are detected
        if "No Fake Messages Found" in fake_message_results or len(fake_message_results) < 5:
            return "✅ No Fake Messages Found", [], 0.0

        # ✅ Extract detected fake messages
        fake_messages = [msg.strip() for msg in fake_message_results.split("\n") if msg.strip()]

        # ✅ Calculate fake message percentage
        fake_message_percentage = (len(fake_messages) / len(filtered_messages)) * 100

        return f"🚨 Fake Messages Detected! ({fake_message_percentage:.2f}% of messages are fake)", fake_messages, fake_message_percentage

    except Exception as e:
        return f"❌ Error detecting fake messages: {e}", [], 0.0
