from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_DIRECTORY = Path(__file__).resolve().parent


# Load the 960 unique reviews
df = pd.read_excel(PROJECT_DIRECTORY / "unique_reviews_for_labeling.xlsx")

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(review):
    review = str(review)

    scores = analyzer.polarity_scores(review)
    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# IMPORTANT:
# Sentiment is determined ONLY from Customer_Opinion_Experience
df["Sentiment"] = df["Customer_Opinion_Experience"].apply(get_sentiment)


# Save the labelled reviews
df.to_excel(
    PROJECT_DIRECTORY / "unique_reviews_labeled.xlsx",
    index=False
)


print("Text-based sentiment labeling completed.")
print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nFile created:")
print("unique_reviews_labeled.xlsx")