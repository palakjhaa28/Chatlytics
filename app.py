import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from parser import parse_chat
from analytics import *
from utils import predict_batch, clean_text, summarize_chat

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.title("📊 WhatsApp Chat Sentiment & Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your WhatsApp chat (.txt file)")

if uploaded_file:
    chat_text = uploaded_file.read().decode("utf-8")
    df = parse_chat(chat_text)

    if df.empty:
        st.error("Couldn't parse any messages. Make sure this is a valid WhatsApp exported .txt file "
                  "(without media, 12-hour time format).")
        st.stop()

    df = df.dropna(subset=["datetime"])

    # ---------------- Date range filter ----------------
    st.subheader("Select Date Range")
    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    start_date, end_date = st.date_input(
        "Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    mask = (df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)
    df = df[mask]

    # ---------------- User filter ----------------
    users = sorted(df["user"].dropna().unique().tolist())
    users.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Analyze for", users)

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    if df.empty:
        st.warning("No messages found for this selection.")
        st.stop()

    # ---------------- Basic stats ----------------
    st.header("Overview")
    total_messages, total_users = basic_stats(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", total_messages)
    col2.metric("Total Users", total_users)
    col3.metric("Total Links Shared", int(df["links"].sum()))

    # ---------------- Messages per user ----------------
    if selected_user == "Overall":
        st.header("Messages per User")
        mpu = messages_per_user(df)
        fig, ax = plt.subplots()
        mpu.plot(kind="bar", ax=ax, color="#25D366")
        ax.set_xlabel("User")
        ax.set_ylabel("Messages")
        st.pyplot(fig)

    # ---------------- Hourly activity ----------------
    st.header("Hourly Activity")
    activity = hourly_activity(df)
    fig, ax = plt.subplots()
    activity.plot(kind="line", marker="o", ax=ax, color="#128C7E")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Messages")
    st.pyplot(fig)

    # ---------------- Emoji analysis ----------------
    st.header("Emoji Usage")
    emoji_stats = emoji_analysis(df)
    if emoji_stats.sum() > 0:
        fig, ax = plt.subplots()
        emoji_stats.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#FFC107")
        ax.set_ylabel("Emoji Count")
        st.pyplot(fig)
    else:
        st.info("No emojis found in this range.")

    # ---------------- Conversation starters ----------------
    if selected_user == "Overall":
        st.header("Conversation Starters")
        starters = conversation_starter(df)
        st.dataframe(starters.rename("Times Started Conversation"))

        # ---------------- Lurker detection ----------------
        st.header("Lurkers (below-average message count)")
        lurkers = lurker_detection(df)
        st.dataframe(lurkers.rename("Message Count"))

    # ---------------- Keywords & WordCloud ----------------
    st.header("Top Keywords")
    messages_list = df["message"].dropna().tolist()
    keywords = extract_keywords(messages_list)
    st.write(", ".join(keywords) if keywords else "Not enough text to extract keywords.")

    cleaned = clean_text(messages_list)
    if cleaned.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(cleaned)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # ---------------- Sentiment analysis ----------------
    st.header("Sentiment Analysis")
    if st.button("Run Sentiment Analysis"):
        with st.spinner("Analyzing sentiment... this may take a moment"):
            texts = df["message"].fillna("").tolist()
            sentiments = predict_batch(texts)
            df["sentiment"] = sentiments

        sentiment_counts = df["sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%")
        st.pyplot(fig)

        if selected_user == "Overall":
            st.subheader("Sentiment by User")
            sentiment_by_user = df.groupby(["user", "sentiment"]).size().unstack(fill_value=0)
            st.dataframe(sentiment_by_user)

    # ---------------- Chat summarization ----------------
    st.header("Chat Summary")
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_chat(df["message"].dropna().tolist())
        st.write(summary if summary.strip() else "Not enough content to summarize.")

else:
    st.info("👆 Upload a WhatsApp chat .txt file to get started.")