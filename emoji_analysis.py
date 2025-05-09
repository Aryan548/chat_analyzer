import pandas as pd
import emoji
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.family'] = 'Segoe UI Emoji'  # Set emoji-compatible font

def extract_emojis(text):
    """Extracts emojis from a given text."""
    return [char for char in text if char in emoji.EMOJI_DATA]

def emoji_analysis(df):
    """Finds the most frequently used emojis and creates a pie chart."""
    if df.empty:
        return None, pd.DataFrame()  # Return empty values if no data

    all_emojis = []
    df['message'].dropna().apply(lambda msg: all_emojis.extend(extract_emojis(str(msg))))

    if not all_emojis:
        return None, pd.DataFrame()  # No emojis found

    emoji_counts = Counter(all_emojis)
    top_emojis = emoji_counts.most_common(10)  # Get top 10 emojis

    if not top_emojis:
        return None, pd.DataFrame()

    labels, sizes = zip(*top_emojis)  # Unpacking labels and sizes

    # ✅ Adjust small slices to reduce overlap
    explode = [0.1 if size < 5 else 0 for size in sizes]  # Only "explode" small slices

    # ✅ Create Pie Chart with Leader Lines & Cleaner Labels
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # 🔥 Hide percentages for slices < 5%
        colors=plt.cm.Paired.colors,
        startangle=140, pctdistance=1.3, labeldistance=1.5, explode=explode
    )

    # ✅ Reduce Font Size for Labels & Percentages
    for text in texts:  
        text.set_fontsize(3)  # Reduce emoji label size  
    for autotext in autotexts:  
        autotext.set_fontsize(1)  # Reduce percentage font size  

    ax.set_title("😃 Emoji Usage Distribution", fontsize=12)  # Reduce title font size

    # ✅ Create a Table of Emojis with Frequency
    emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Count'])

    return fig, emoji_df
