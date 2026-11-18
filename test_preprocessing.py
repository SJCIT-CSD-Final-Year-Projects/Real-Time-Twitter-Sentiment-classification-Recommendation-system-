import unittest

from preprocessing import clean_text, get_sentiment


class PreprocessingTests(unittest.TestCase):
	def test_clean_text_removes_url_and_punctuation(self):
		self.assertEqual(
			clean_text("LOVE this! https://example.com #EV"),
			"love this ev",
		)

	def test_clean_text_handles_missing_values(self):
		self.assertEqual(clean_text(None), "")

	def test_word_matching_uses_token_boundaries(self):
		self.assertEqual(
			get_sentiment({
				"Rating_1_to_5": 3,
				"Customer_Opinion_Experience": "goodbye",
			}),
			"Neutral",
		)

	def test_negated_positive_does_not_become_positive(self):
		self.assertNotEqual(
			get_sentiment({
				"Rating_1_to_5": 5,
				"Customer_Opinion_Experience": "not good",
			}),
			"Positive",
		)

	def test_missing_rating_does_not_raise(self):
		self.assertEqual(
			get_sentiment({
				"Rating_1_to_5": None,
				"Customer_Opinion_Experience": "great excellent",
			}),
			"Positive",
		)


if __name__ == "__main__":
	unittest.main()