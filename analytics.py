import pandas as pd
import emoji
from collections import Counter
import re

def extract_keywords(messages):

    words = []

    for msg in messages:
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', msg.lower())
        words.extend(tokens)

    common_words = Counter(words).most_common(10)

    return [w[0] for w in common_words]

def basic_stats(df):

    total_messages = df.shape[0]
    total_users = df["user"].nunique()

    return total_messages, total_users


def messages_per_user(df):

    return df["user"].value_counts()


def hourly_activity(df):

    return df.groupby("hour").size()


def emoji_count(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)


def emoji_analysis(df):

    df["emoji_count"] = df["message"].apply(emoji_count)

    return df.groupby("user")["emoji_count"].sum()


def conversation_starter(df):

    df["prev_time"] = df["datetime"].shift(1)
    df["starter"] = (df["datetime"] - df["prev_time"]).dt.seconds > 3600

    return df[df["starter"]]["user"].value_counts()


def lurker_detection(df):

    user_msg_counts = df["user"].value_counts()
    lurkers = user_msg_counts[user_msg_counts < user_msg_counts.mean()]

    return lurkers