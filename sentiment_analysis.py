"""Sentiment classification for customer review text."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

_analyzer = SentimentIntensityAnalyzer()
SENTIMENT_LABELS = {"Positive", "Neutral", "Negative"}


def train_sentiment_models(training_data):
    """Fit the project's classifiers on unique, labeled review text."""
    required_columns = {"Customer_Opinion_Experience", "Sentiment"}
    missing_columns = required_columns.difference(training_data.columns)
    if missing_columns:
        raise ValueError(
            "Sentiment training data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    training_reviews = training_data[
        ["Customer_Opinion_Experience", "Sentiment"]
    ].copy()
    training_reviews["Customer_Opinion_Experience"] = (
        training_reviews["Customer_Opinion_Experience"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    training_reviews["Sentiment"] = (
        training_reviews["Sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    training_reviews = training_reviews[
        training_reviews["Customer_Opinion_Experience"].ne("")
        & training_reviews["Sentiment"].isin(SENTIMENT_LABELS)
    ].drop_duplicates(subset=["Customer_Opinion_Experience"])

    if training_reviews["Sentiment"].nunique() < 2:
        raise ValueError("Sentiment training data must contain at least two labels.")

    training_reviews, holdout_reviews = train_test_split(
        training_reviews,
        test_size=0.2,
        random_state=42,
        stratify=training_reviews["Sentiment"],
    )

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    class_counts = training_reviews["Sentiment"].value_counts()
    sample_weights = training_reviews["Sentiment"].map(
        lambda label: len(training_reviews)
        / (len(class_counts) * class_counts[label])
    )
    models = {}

    for name, classifier in classifiers.items():
        pipeline = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000,
                sublinear_tf=True,
            ),
            classifier,
        )
        classifier_step = pipeline.steps[-1][0]
        pipeline.fit(
            training_reviews["Customer_Opinion_Experience"],
            training_reviews["Sentiment"],
            **{f"{classifier_step}__sample_weight": sample_weights},
        )
        models[name] = pipeline

    return models, training_reviews, holdout_reviews


def analyze_sentiment(texts):
    """Classify each review and return aligned scores and labels."""
    scores = []
    labels = []

    for text in texts:
        clean_text = "" if text is None else str(text).strip()
        score = _analyzer.polarity_scores(clean_text)["compound"]
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        scores.append(score)
        labels.append(label)

    return scores, labels
