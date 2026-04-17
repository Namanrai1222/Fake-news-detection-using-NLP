import unittest

from src.preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):
	def test_non_string_returns_empty_string(self):
		self.assertEqual(clean_text(None), "")
		self.assertEqual(clean_text(123), "")

	def test_removes_source_tags_and_noise(self):
		text = "WASHINGTON (Reuters) - [Breaking] This IS only a sample report with punctuation!!!"
		cleaned = clean_text(text)

		self.assertNotIn("reuters", cleaned)
		self.assertNotIn("breaking", cleaned)
		self.assertNotIn("!", cleaned)
		self.assertIn("sample", cleaned)
		self.assertIn("report", cleaned)

	def test_reduces_common_stopwords(self):
		text = "This is the news that we are reading in the city today"
		cleaned = clean_text(text)
		tokens = cleaned.split()

		self.assertNotIn("the", tokens)
		self.assertNotIn("is", tokens)
		self.assertIn("news", tokens)


if __name__ == "__main__":
	unittest.main()
