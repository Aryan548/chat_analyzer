import pandas as pd
import re

def preprocess(data):
    messages = []
    dates = []
    users = []

    # ✅ Updated regex pattern to support different WhatsApp formats (12-hour and 24-hour)
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},? \s?\d{1,2}:\d{2}\s?[APMampm]*?) - (.*?): (.+)'

    for line in data.split("\n"):
        match = re.match(pattern, line)
        if match:
            date, user, message = match.groups()
            dates.append(date.strip())
            users.append(user.strip())
            messages.append(message.strip())
        elif messages:
            # ✅ Handle multiline messages correctly
            messages[-1] += " " + line.strip()

    # ✅ Return empty DataFrame if no valid messages are found
    if not dates or not messages:
        return pd.DataFrame(columns=["date", "user", "message", "month"])

    df = pd.DataFrame({"date": dates, "user": users, "message": messages})

    # ✅ Convert date to datetime format, dropping invalid entries
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # ✅ New Feature: Extract month for Peak Chat Month Analysis
    df["month"] = df["date"].dt.strftime("%B")  # Extracts month name (e.g., January)

    return df
