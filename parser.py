import re
import pandas as pd
import emoji


def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)


def parse_chat(file):

    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.*?): (.*)'

    data = []

    for line in file.split("\n"):

        match = re.match(pattern, line)

        if match:
            date, time, user, message = match.groups()
            data.append([date, time, user, message])

    df = pd.DataFrame(data, columns=["date", "time", "user", "message"])

    # create datetime
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed",errors="coerce")

    # time features
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()

    # emoji count
    df["emoji_count"] = df["message"].apply(count_emojis)

    # link detection
    df["links"] = df["message"].str.contains(r"http[s]?://", regex=True)

    return df