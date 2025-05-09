import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(df):
    """Analyzes sentiment of each message and classifies it into categories."""
    if 'message' not in df.columns or df.empty:
        return df

    # Compute sentiment scores
    df['sentiment_score'] = df['message'].apply(lambda msg: sia.polarity_scores(str(msg))['compound'])

    # ✅ Expanded Sentiment Categories
    def categorize_sentiment(score):
        if score <= -0.6:
            return "Very Negative"
        elif -0.6 < score <= -0.2:
            return "Negative"
        elif -0.2 < score <= 0.2:
            return "Neutral"
        elif 0.2 < score <= 0.6:
            return "Positive"
        else:
            return "Very Positive"

    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    return df  # ✅ Returns the DataFrame with sentiment labels

def plot_sentiment_distribution(df):
    """Creates a bar graph for sentiment distribution."""
    if df.empty or 'sentiment_category' not in df.columns:
        return None

    sentiment_counts = df['sentiment_category'].value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))  # ✅ Keep graph small
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)

    ax.set_xlabel("Sentiment Category", fontsize=10, labelpad=10)  # ✅ Adjust font and padding
    ax.set_ylabel("Message Count", fontsize=10, labelpad=10)
    ax.set_title("📊 Sentiment Distribution", fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)  # ✅ Rotate to avoid overlap

    return fig  # ✅ Returns the figure for Streamlit
