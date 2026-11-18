from pathlib import Path

import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent


# Load the real dataset
df = pd.read_excel(
    PROJECT_DIRECTORY / 'Ola_Scooter_Customer_Experience_Dataset_60000.xlsx',
    sheet_name='All_Reviews_60000'
)

print("Total records:", len(df))


# Get unique customer reviews
unique_reviews = (
    df[['Customer_Opinion_Experience']]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)


print("Unique reviews:", len(unique_reviews))


# Add an empty sentiment column
unique_reviews['Sentiment'] = ''


# Save the reviews for labeling
unique_reviews.to_excel(
    PROJECT_DIRECTORY / 'unique_reviews_for_labeling.xlsx',
    index=False
)


print("\nFile created successfully:")
print("unique_reviews_for_labeling.xlsx")
print("\nNow you have to assign:")
print("Positive")
print("Neutral")
print("Negative")