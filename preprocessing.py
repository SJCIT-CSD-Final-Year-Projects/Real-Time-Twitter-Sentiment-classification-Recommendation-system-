from pathlib import Path
import re

import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIRECTORY / "Ola_Scooter_Customer_Experience_Dataset_60000.xlsx"
SHEET_NAME = "All_Reviews_60000"

POSITIVE_WORDS = {
    "good", "great", "excellent", "smooth", "comfortable", "amazing",
    "satisfied", "happy", "reliable", "love", "best", "easy", "perfect",
}
NEGATIVE_WORDS = {
    "bad", "poor", "worst", "problem", "issue", "disappointing",
    "difficult", "uncomfortable", "expensive", "terrible", "slow",
    "failure", "worse", "unsafe",
}
NEGATION_WORDS = {"not", "no", "never", "hardly", "barely"}


def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_sentiment(row):
    rating = pd.to_numeric(row.get("Rating_1_to_5"), errors="coerce")
    score = 0

    if pd.notna(rating):
        if rating >= 4:
            score += 2
        elif rating < 3:
            score -= 2

    review = str(row.get("Customer_Opinion_Experience", "")).lower()
    review = re.sub(r"\b(can't|cannot)\b", "can not", review)
    review = re.sub(r"\bwon't\b", "will not", review)
    review = re.sub(r"\b(\w+)n't\b", r"\1 not", review)
    tokens = re.findall(r"[a-z]+", review)

    for index, word in enumerate(tokens):
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue

        polarity = 1 if word in POSITIVE_WORDS else -1
        preceding_tokens = tokens[max(0, index - 3):index]
        if any(token in NEGATION_WORDS for token in preceding_tokens):
            polarity *= -1
        score += polarity

    if score >= 2:
        return "Positive"
    if score <= -2:
        return "Negative"
    return "Neutral"


def main():
    data = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    print("Dataset loaded successfully.")
    print("Total records:", len(data))

    data["Cleaned_Review"] = data[
        "Customer_Opinion_Experience"
    ].apply(clean_text)
    data["Sentiment"] = data.apply(get_sentiment, axis=1)

    if not data.empty:
        print("\nOriginal review:")
        print(data["Customer_Opinion_Experience"].iloc[0])
        print("\nCleaned review:")
        print(data["Cleaned_Review"].iloc[0])
        print("\nSentiment:")
        print(data["Sentiment"].iloc[0])

    print("\nSentiment distribution:")
    print(data["Sentiment"].value_counts())
    print("\nFirst five records:")
    print(
        data[
            [
                "Model",
                "Experience_Area",
                "Rating_1_to_5",
                "Daily_Usage",
                "Ownership_Duration",
                "Sentiment",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()