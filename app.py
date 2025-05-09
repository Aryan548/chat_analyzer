import streamlit as st
import pandas as pd
import preprocessor
import helper
import sentiment_analysis
import emoji_analysis
import summarizer
import hate_speech  # Import Hate Speech Detection
import fake_message_detector  # Import Fake Message Detection
import chatbot  # Import Chatbot for Q&A

# Streamlit Page Config - MUST BE FIRST
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

# Apply global font styling
st.markdown("""
    <style>
        * {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.markdown(
    "<h1 style='text-align: center; color: #00b4d8;'>"
    "<img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg' width='70' style='vertical-align: middle; margin-right: 20px;'>"
    "WhatsApp Chat Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown("<h3 style='text-align: center;'>📊 Unlock insights from your chats!</h3>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("📁 Upload a WhatsApp chat file (.txt)", type="txt")

if uploaded_file is not None:
    try:
        # Read and Preprocess Data
        data = uploaded_file.getvalue().decode("utf-8")
        df = preprocessor.preprocess(data)

        if df.empty:
            st.warning("⚠️ No valid messages found! Check your file format and try again.")
        else:
            st.success("✅ Chat file successfully processed!")

            # Chat Statistics
            st.header("📊 Chat Statistics")
            col1, col2, col3, col4 = st.columns(4)
            num_messages, num_words, num_media, num_links = helper.fetch_stats(df)
            col1.metric("📝 Total Messages", num_messages)
            col2.metric("🔤 Total Words", num_words)
            col3.metric("📷 Media Shared", num_media)
            col4.metric("🔗 Links Shared", num_links)

            # Show Processed Chat Data
            if st.checkbox("🔍 Show Processed Chat Data"):
                st.dataframe(df)

            # Most Active Users
            st.header("🏆 Most Active Users")
            user_stats, user_percentages = helper.active_users(df)
            if not user_stats.empty:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(user_stats)
                with col2:
                    st.dataframe(user_percentages)

            # Word Cloud
            st.header("☁️ Word Cloud")
            wc = helper.create_wordcloud(df)
            st.image(wc, caption="Most Used Words", use_container_width=True, width=400)

            # Common Topics
            st.header("📝 Common Topics")
            common_words_df = helper.most_common_words(df, num_words=15)
            st.table(common_words_df)

            # Peak Chat Hours & Peak Chat Month Analysis
            st.header("⏰ Peak Chat Hours & 📅 Peak Chat Month Analysis")
            col1, col2 = st.columns(2)
            with col1:
                peak_hours_fig = helper.peak_chat_hours(df)
                if peak_hours_fig:
                    st.pyplot(peak_hours_fig)
            with col2:
                peak_month_fig = helper.peak_chat_month(df)
                if peak_month_fig:
                    st.pyplot(peak_month_fig)

            # Sentiment & Emoji Analysis
            st.header("😊 Sentiment Analysis & 😃 Emoji Analysis (Pie Chart)")
            col1, col2 = st.columns(2)
            with col1:
                sentiment_df = sentiment_analysis.analyze_sentiment(df)
                sentiment_fig = sentiment_analysis.plot_sentiment_distribution(sentiment_df)
                if sentiment_fig:
                    fig_sentiment = sentiment_fig.get_figure()
                    fig_sentiment.set_size_inches(5, 4)
                    st.pyplot(fig_sentiment)
            with col2:
                emoji_pie_chart = helper.emoji_pie_chart(df)
                if emoji_pie_chart:
                    fig_emoji = emoji_pie_chart.get_figure()
                    fig_emoji.set_size_inches(5, 2)
                    st.pyplot(fig_emoji)

            # Chat Summary
            st.header("📝 Chat Summary")
            summary = summarizer.summarize_chat(df)
            if "❌ Error" in summary:
                st.error(summary)
            else:
                st.write(f"✍️ **Summary:** {summary}")

            # Hate Speech Detection
            st.header("🚨 Hate Speech Detection")
            hate_speech_result, hate_messages = hate_speech.detect_hate_speech(df)
            if hate_messages:
                st.error("⚠️ Hate Speech Found in Messages:")
                for user, msg in hate_messages:
                    st.write(f"❌ **{user}:** {msg}")
            else:
                st.success(hate_speech_result)

            # Fake Message Detection
            st.header("🔍 Fake Message Detection")
            fake_result, fake_messages, fake_percentage = fake_message_detector.detect_fake_messages(df)
            if fake_messages:
                st.error(f"⚠️ Fake Messages Found ({fake_percentage:.2f}% of messages are fake)")
                for entry in fake_messages:
                    if ":" in entry:
                        sender, message = entry.split(":", 1)
                        st.write(f"❌ **{sender.strip()}:** {message.strip()}")
                    else:
                        st.write(f"❌ {entry}")
            else:
                st.success(fake_result)

            # Chatbot Q&A
            
            st.header("🤖 Chatbot - Ask Questions on Your Chat Data")
            user_query = st.text_input("💬 Ask a question about the chat:")
            if st.button("🔍 Get Answer"):
                if user_query:
                    response = chatbot.answer_query(df, user_query)
                    st.write(f"🤖 **Chatbot:** {response}")
                else:
                    st.warning("⚠️ Please enter a question to proceed.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
