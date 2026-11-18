import unittest

import pandas as pd

from sentiment_analysis import train_sentiment_models


class SentimentModelTests(unittest.TestCase):
    def test_trains_all_classifiers_and_returns_valid_labels(self):
        training_data = pd.DataFrame(
            [
                {"Customer_Opinion_Experience": "great battery range", "Sentiment": "Positive"},
                {"Customer_Opinion_Experience": "excellent battery range", "Sentiment": "Positive"},
                {"Customer_Opinion_Experience": "great reliable scooter", "Sentiment": "Positive"},
                {"Customer_Opinion_Experience": "amazing scooter range", "Sentiment": "Positive"},
                {"Customer_Opinion_Experience": "smooth and reliable", "Sentiment": "Positive"},
                {"Customer_Opinion_Experience": "poor battery range", "Sentiment": "Negative"},
                {"Customer_Opinion_Experience": "terrible service failure", "Sentiment": "Negative"},
                {"Customer_Opinion_Experience": "poor unreliable scooter", "Sentiment": "Negative"},
                {"Customer_Opinion_Experience": "bad service problem", "Sentiment": "Negative"},
                {"Customer_Opinion_Experience": "unsafe and disappointing", "Sentiment": "Negative"},
                {"Customer_Opinion_Experience": "average battery range", "Sentiment": "Neutral"},
                {"Customer_Opinion_Experience": "okay service experience", "Sentiment": "Neutral"},
                {"Customer_Opinion_Experience": "average scooter ride", "Sentiment": "Neutral"},
                {"Customer_Opinion_Experience": "ordinary daily ride", "Sentiment": "Neutral"},
                {"Customer_Opinion_Experience": "it is acceptable", "Sentiment": "Neutral"},
            ]
        )

        models, training_reviews, holdout_reviews = train_sentiment_models(training_data)
        predictions = [
            model.predict(["great battery", "poor service"])
            for model in models.values()
        ]

        self.assertEqual(
            set(models),
            {
                "Naive Bayes",
                "Logistic Regression",
                "Linear SVM",
                "Random Forest",
                "Decision Tree",
            },
        )
        self.assertEqual(len(holdout_reviews), 3)
        self.assertTrue(
            set(holdout_reviews["Sentiment"]) == {"Positive", "Neutral", "Negative"}
        )
        self.assertTrue(
            set(holdout_reviews["Customer_Opinion_Experience"]).isdisjoint(
                set(training_reviews["Customer_Opinion_Experience"])
            )
        )
        for model_predictions in predictions:
            self.assertEqual(len(model_predictions), 2)
            self.assertTrue(
                set(model_predictions).issubset({"Positive", "Neutral", "Negative"})
            )


if __name__ == "__main__":
    unittest.main()