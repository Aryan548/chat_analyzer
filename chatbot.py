import google.generativeai as genai
import os
from dotenv import load_dotenv

# ✅ Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def answer_query(df, user_query):
    """Uses Google Gemini AI to answer user queries based on chat data."""

    if df.empty or "message" not in df.columns or "user" not in df.columns:
        return "No chat data available for answering questions."

    # ✅ Combine user and message for more context
    df["combined"] = df["user"].astype(str) + ": " + df["message"].astype(str)
    chat_lines = df["combined"].dropna().tolist()

    # ✅ Limit to the last 200 messages to stay within API context size
    limited_chat = "\n".join(chat_lines[-200:])

    if len(limited_chat.strip()) == 0:
        return "No valid messages found in the chat for answering."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are a smart chatbot analyzing a WhatsApp group chat. "
            "Try to answer the user's question using the information from the chat below. "
            "If the chat doesn't directly provide the answer, use your best reasoning to guess based on context.\n\n"
            f"Chat:\n{limited_chat}\n\n"
            f"User Question: {user_query}"
        )

        response = model.generate_content(prompt)
        chatbot_response = response.text.strip()

        # ✅ Even if vague, always return something
        if len(chatbot_response) < 3:
            return "Based on the chat, it's a bit unclear, but here's what I can infer: 🤔 ..."

        return chatbot_response

    except Exception as e:
        return f"❌ Error in chatbot response: {e}"
