from pathlib import Path

import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent

# Load the original 60,000-review dataset
df = pd.read_excel(
    PROJECT_DIRECTORY / 'Ola_Scooter_Customer_Experience_Dataset_60000.xlsx',
    sheet_name='All_Reviews_60000'
)

# Load the file containing the labelled unique reviews
labels = pd.read_excel(
    PROJECT_DIRECTORY / 'unique_reviews_labeled.xlsx'
)

# Make sure the review column names are the same
print("Original columns:")
print(df.columns.tolist())

print("\nLabel file columns:")
print(labels.columns.tolist())

# Create a clean matching column in both files
df['Review_Key'] = (
    df['Customer_Opinion_Experience']
    .fillna('')
    .astype(str)
    .str.strip()
    .str.lower()
)

labels['Review_Key'] = (
    labels['Customer_Opinion_Experience']
    .fillna('')
    .astype(str)
    .str.strip()
    .str.lower()
)

# Keep only the review and its sentiment label
label_mapping = labels[['Review_Key', 'Sentiment']].drop_duplicates(
    subset='Review_Key'
)

# Map sentiment labels to all 60,000 reviews
df['Sentiment'] = df['Review_Key'].map(
    label_mapping.set_index('Review_Key')['Sentiment']
)

# Remove temporary column
df.drop(columns=['Review_Key'], inplace=True)

# Check results
print("\nTotal records:", len(df))
print("Labels found:", df['Sentiment'].notna().sum())
print("Labels missing:", df['Sentiment'].isna().sum())

print("\nSentiment distribution:")
print(df['Sentiment'].value_counts(dropna=False))

# Save the labelled 60,000-row dataset
df.to_excel(
    PROJECT_DIRECTORY / 'Ola_60000_Labelled.xlsx',
    index=False
)

print("\nSuccessfully created:")
print(PROJECT_DIRECTORY / "Ola_60000_Labelled.xlsx")