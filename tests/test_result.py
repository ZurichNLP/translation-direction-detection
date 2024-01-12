from unittest import TestCase

from translation_direction_detection.detector import TranslationDirectionResult


class TranslationDirectionResultTestCase(TestCase):

    def setUp(self):
        self.sent_result = TranslationDirectionResult(
            sentence1="This is a test.",
            sentence2="Dies ist ein Test.",
            lang1="en",
            lang2="de",
            raw_prob_1_to_2=0.6,
            raw_prob_2_to_1=0.4,
        )
        self.doc_result = TranslationDirectionResult(
            sentence1=["This is a test.", "This is another test."],
            sentence2=["Dies ist ein Test.", "Dies ist ein weiterer Test."],
            lang1="en",
            lang2="de",
            raw_prob_1_to_2=0.45,
            raw_prob_2_to_1=0.55,
            pvalue=0.05,
        )

    def test_num_sentences(self):
        self.assertEqual(self.sent_result.num_sentences, 1)
        self.assertEqual(self.doc_result.num_sentences, 2)

    def test_predicted_direction(self):
        self.assertEqual(self.sent_result.predicted_direction, "en→de")
        self.assertEqual(self.doc_result.predicted_direction, "de→en")

    def test_str(self):
        sent_str = str(self.sent_result)
        self.assertIn("Predicted direction: en→de", sent_str)
        self.assertIn("1 sentence pair", sent_str)
        self.assertNotIn("1 sentence pairs", sent_str)
        self.assertIn("en→de: 0.550", sent_str)
        self.assertIn("de→en: 0.450", sent_str)
        self.assertNotIn("p-value", sent_str)
        doc_str = str(self.doc_result)
        self.assertIn("Predicted direction: de→en", doc_str)
        self.assertIn("2 sentence pairs", doc_str)
        self.assertIn("en→de: 0.475", doc_str)
        self.assertIn("de→en: 0.525", doc_str)
        self.assertIn("p-value: 0.05", doc_str)
