import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def detect_hate_speech(df):
    """Detects hate speech in WhatsApp messages using Google Gemini API and returns flagged messages along with senders."""

    if df.empty or "message" not in df.columns or "user" not in df.columns:
        return "No chat data available for analysis.", []

    # Extract messages and users, while removing "<Media omitted>" lines
    filtered_messages = df[df["message"].str.contains("<Media omitted>") == False].dropna(subset=["message"])
    
    if filtered_messages.empty:
        return "No valid messages to analyze.", []

    # Convert messages into a structured text format (User: Message)
    chat_text = "\n".join(f"{row['user']}: {row['message']}" for _, row in filtered_messages.iterrows())

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Use correct model
        response = model.generate_content(
            f"Below is a WhatsApp chat conversation. Identify and list any messages containing hate speech, along with the sender's name. "
            "If no hate speech is found, respond with 'No Hate Speech Found'.\n\n"
            "Chat Messages:\n" + chat_text
        )

        hate_speech_results = response.text.strip()

        # ✅ If API response contains "No Hate Speech Found", return success message
        if "No Hate Speech Found" in hate_speech_results or len(hate_speech_results) < 5:
            return "✅ No Hate Speech Found", []

        # ✅ Otherwise, extract detected hate speech messages along with senders
        hate_messages = []
        for line in hate_speech_results.split("\n"):
            if ":" in line:  # Ensure the line has "User: Message" format
                user, message = line.split(":", 1)
                hate_messages.append((user.strip(), message.strip()))  # ✅ Store sender and message

        return "🚨 Hate Speech Detected", hate_messages[:5]  # ✅ Limit output to first 5 messages  
 

    except Exception as e:
        return f"❌ Error detecting hate speech: {e}", []
