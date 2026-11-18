from pathlib import Path

import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
df = pd.read_excel(PROJECT_DIRECTORY / "unique_reviews_labeled.xlsx")

for sentiment in ["Positive", "Negative", "Neutral"]:
    print("\n" + "=" * 70)
    print(sentiment)
    print("=" * 70)

    samples = df[df["Sentiment"] == sentiment].sample(
        min(10, len(df[df["Sentiment"] == sentiment])),
        random_state=42
    )

    for i, row in samples.iterrows():
        print("\nReview:", row["Customer_Opinion_Experience"])
        print("Label:", row["Sentiment"])