import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# ✅ Function 1: Fetch Chat Statistics
def fetch_stats(df):
    """Computes chat statistics like total messages, words, media, and links."""
    if df.empty:
        return 0, 0, 0, 0

    num_messages = df.shape[0]
    num_words = df['message'].apply(lambda x: len(str(x).split())).sum()
    num_media = df[df['message'].str.lower() == '<media omitted>'].shape[0]
    num_links = df['message'].apply(lambda x: len(re.findall(r'http\S+', str(x)))).sum()

    return num_messages, num_words, num_media, num_links

# ✅ Function 2: Find Most Active Users
def active_users(df):
    """Finds the most active users in the chat."""
    if df.empty or 'user' not in df.columns:
        return pd.Series(), pd.Series()

    user_counts = df['user'].value_counts().head(10)
    user_percentages = round((df['user'].value_counts(normalize=True) * 100), 2)
    
    return user_counts, user_percentages

# ✅ Function 3: Generate Word Cloud (Smaller Size)
def create_wordcloud(df):
    """Generates a word cloud from chat messages."""
    if df.empty:
        return None

    text = " ".join(str(msg) for msg in df['message'] if msg)
    wordcloud = WordCloud(width=600, height=250, background_color="black").generate(text)
    
    return wordcloud.to_image()

# ✅ Function 4: Find Most Common Words (Top 15)
def most_common_words(df, num_words=15):
    """Finds the most frequently used words in chat."""
    if df.empty:
        return pd.DataFrame()

    words = " ".join(df['message']).lower().split()

    # Load stop words (if available)
    try:
        stop_words = set(open("stop_words.txt").read().split())
    except FileNotFoundError:
        stop_words = set()

    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(filtered_words)

    return pd.DataFrame(word_freq.most_common(num_words), columns=['Word', 'Count'])

# ✅ Function 5: Peak Chat Hours Analysis (Smaller Graph)
def peak_chat_hours(df):
    """Analyzes peak chat hours by plotting a histogram of message timestamps."""
    if df.empty or 'date' not in df.columns:
        return None

    df['hour'] = df['date'].dt.hour

    plt.figure(figsize=(6, 4))  # Adjust size
    sns.histplot(df['hour'], bins=24, kde=False, color='purple')
    plt.xlabel("Hour of the Day")
    plt.ylabel("Message Count")
    plt.title("⏰ Peak Chat Hours")
    plt.xticks(range(0, 24), rotation=45)  # ✅ Rotate labels for better spacing

    return plt


# ✅ Function 6: Peak Chat Month Analysis (NEW FEATURE 🚀)
def peak_chat_month(df):
    """Analyzes peak chat months and ensures correct order from January to December."""
    if df.empty or 'month' not in df.columns:
        return None

    # Define proper month order
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]

    # Ensure 'month' column is a categorical type with correct order
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    # Count messages per month in correct order
    month_counts = df['month'].value_counts().sort_index()

    # Plot the graph
    plt.figure(figsize=(8, 5))
    sns.barplot(x=month_counts.index, y=month_counts.values, palette="Blues_r")
    plt.xlabel("Month")
    plt.ylabel("Message Count")
    plt.title("📆 Peak Chat Month")
    plt.xticks(rotation=45)  # Rotate for better readability

    return plt



# ✅ Function 7: Emoji Analysis (Pie Chart, Smaller Size)
def emoji_pie_chart(df):
    """Creates a pie chart for emoji usage in chats (smaller size)."""
    if df.empty:
        return None

    all_emojis = []
    df['message'].dropna().apply(lambda msg: all_emojis.extend([char for char in str(msg) if char in emoji.EMOJI_DATA]))

    emoji_counts = Counter(all_emojis)
    top_emojis = emoji_counts.most_common(10)

    if not top_emojis:
        return None

    labels, sizes = zip(*top_emojis)

    fig, ax = plt.subplots(figsize=(3, 3))  # ✅ Thread-safe Fix
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("pastel"), textprops={'fontsize': 7})
    ax.set_title("😃 Emoji Usage Distribution", fontsize=12)
    
    return fig

# ✅ Function 8: Sentiment Analysis Visualization (Fixed)
def sentiment_analysis(df):
    """Plots a histogram of sentiment scores with expanded categories."""
    if df.empty or 'sentiment' not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))  # ✅ Thread-safe Fix
    sns.histplot(df['sentiment'], bins=20, kde=True, color='blue', ax=ax)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Message Count")
    ax.set_title("Sentiment Distribution")

    return fig
# ✅ Function: Sentiment Analysis Bar Chart
def sentiment_analysis_bar(df):
    """Generates a sentiment distribution bar graph."""
    if df.empty or 'sentiment_category' not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))  # ✅ Smaller graph
    sns.countplot(data=df, x='sentiment_category', palette="coolwarm", ax=ax)
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Message Count")
    ax.set_title("📊 Sentiment Analysis")

    return fig  # ✅ Returns the figure

