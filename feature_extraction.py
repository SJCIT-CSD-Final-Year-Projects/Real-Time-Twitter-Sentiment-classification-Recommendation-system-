from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(text_data):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    features = vectorizer.fit_transform(text_data)

    return features, vectorizer