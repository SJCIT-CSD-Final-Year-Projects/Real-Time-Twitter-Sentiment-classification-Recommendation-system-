import unittest

from scipy.sparse import issparse

from feature_extraction import create_tfidf_features


class TfidfFeatureTests(unittest.TestCase):
    def test_vectorizer_returns_sparse_features_and_reusable_model(self):
        reviews = [
            "great battery range",
            "poor battery service",
            "smooth ride and great range",
        ]

        features, vectorizer = create_tfidf_features(reviews)
        transformed = vectorizer.transform(["great range"])

        self.assertTrue(issparse(features))
        self.assertEqual(features.shape[0], len(reviews))
        self.assertEqual(transformed.shape[0], 1)
        self.assertEqual(transformed.shape[1], features.shape[1])


if __name__ == "__main__":
    unittest.main()