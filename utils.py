import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

reverse_label_map = {
    0: "Happy",
    1: "Sad",
    2: "Depressed",
    3: "Neutral",
    4: "Excitement"
}

def predict_batch(texts, batch_size=64):

    predictions = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())

    return [reverse_label_map[p] for p in predictions]


# -----------------------------
# CHAT SUMMARIZATION MODEL
# -----------------------------



import streamlit as st


@st.cache_resource
def load_summarizer():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

summarizer = load_summarizer()

summarizer = load_summarizer()

import re

def clean_text(messages):

    text = " ".join(messages)

    # remove media messages
    text = re.sub(r"<Media omitted>", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


from collections import Counter


def summarize_chat(messages):

    text = " ".join(messages)

    # remove media messages
    text = re.sub(r"<Media omitted>", "", text)

    # split sentences based on line breaks
    sentences = text.split("\n")

    words = re.findall(r'\w+', text.lower())

    word_freq = Counter(words)

    sentence_scores = {}

    for sentence in sentences:
        for word in sentence.lower().split():
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    # pick top 3 sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

    summary = " ".join(summary_sentences)

    return summary