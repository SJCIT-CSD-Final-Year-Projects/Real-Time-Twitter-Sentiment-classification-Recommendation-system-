import unittest

import pandas as pd

from recommendation import get_personalized_recommendation


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.reviews = pd.DataFrame(
            [
                {"Model": "Model A", "Experience_Area": "Battery", "Sentiment": "Positive"},
                {"Model": "Model A", "Experience_Area": "Ride", "Sentiment": "Negative"},
                {"Model": "Model B", "Experience_Area": "Battery", "Sentiment": "Positive"},
                {"Model": "Model B", "Experience_Area": "Ride", "Sentiment": "Positive"},
            ]
        )

    def test_ranks_models_using_equal_area_weight(self):
        best_model, results = get_personalized_recommendation(
            self.reviews,
            ["Battery", "Ride"],
        )

        self.assertEqual(best_model["Model"], "Model B")
        self.assertEqual(results.iloc[0]["Recommendation"], "Highly Recommended")
        self.assertEqual(results.iloc[0]["Reviews"], 2)

    def test_empty_selection_returns_no_recommendation(self):
        best_model, results = get_personalized_recommendation(
            self.reviews,
            [],
        )

        self.assertIsNone(best_model)
        self.assertTrue(results.empty)


if __name__ == "__main__":
    unittest.main()