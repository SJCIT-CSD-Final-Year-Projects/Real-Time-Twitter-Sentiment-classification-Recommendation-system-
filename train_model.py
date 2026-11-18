from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# 1. LOAD LABELLED DATASET
# ============================================================

DATA_FILE = Path(__file__).resolve().parent / "Ola_60000_Labelled.xlsx"
df = pd.read_excel(DATA_FILE)

print("Total reviews:", len(df))
print("\nExperience Areas:")
print(df["Experience_Area"].value_counts())



# ============================================================
# 2. KEEP ONLY UNIQUE REVIEWS
# ============================================================

unique_df = df.drop_duplicates(
    subset=["Customer_Opinion_Experience"]
).copy()

print("Unique reviews:", len(unique_df))


# ============================================================
# 3. GET REVIEW TEXT AND SENTIMENT LABELS
# ============================================================

reviews = (
    unique_df["Customer_Opinion_Experience"]
    .fillna("")
    .astype(str)
)

y = unique_df["Sentiment"]


# ============================================================
# 4. SPLIT UNIQUE REVIEWS
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    reviews,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train_text))
print("Testing reviews:", len(X_test_text))


# ============================================================
# 5. TF-IDF FEATURE EXTRACTION
# ============================================================

tfidf_options = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 20000,
    "stop_words": "english",
    "sublinear_tf": True,
}
tfidf = TfidfVectorizer(**tfidf_options)

# Learn TF-IDF only from training reviews
X_train = tfidf.fit_transform(X_train_text)

# Transform test reviews using the same TF-IDF
X_test = tfidf.transform(X_test_text)

print("TF-IDF features:", X_train.shape[1])


# ============================================================
# 6. DEFINE FIVE ML MODELS
# ============================================================

models = {

    "Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "SVM":
        LinearSVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        )
}


# ============================================================
# 7. TRAIN AND TEST EACH MODEL
# ============================================================

print("\n===== TEST SET RESULTS =====")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )
    print(
        f"Macro F1: {macro_f1 * 100:.2f}%"
    )

    # Precision, Recall and F1-score
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


# ============================================================
# 8. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n\n===== 5-FOLD CROSS-VALIDATION =====")

for name, model in models.items():

    pipeline = make_pipeline(
        TfidfVectorizer(**tfidf_options),
        model,
    )
    scores = cross_val_score(
        pipeline,
        X_train_text,
        y_train,
        cv=5,
        scoring="f1_macro"
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Fold macro F1:",
        scores
    )

    print(
        f"Average macro F1: {scores.mean() * 100:.2f}%"
    )