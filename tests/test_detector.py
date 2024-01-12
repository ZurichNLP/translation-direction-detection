from unittest import TestCase

from translation_direction_detection.detector import TranslationDirectionDetector


class SentenceLevelTranslationDirectionDetectorTestCase(TestCase):

    def setUp(self):
        self.detector = TranslationDirectionDetector()

    def test_sentence(self):
        sentence1 = "Können Sie mir dabei weiter helfen?"
        sentence2 = "Pouvez-vous m'aider ?"
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentence1, sentence2, lang1, lang2)
        self.assertEqual(result.predicted_direction, "de→fr")
        self.assertEqual(result.sentence1, sentence1)
        self.assertEqual(result.sentence2, sentence2)
        self.assertEqual(result.lang1, lang1)
        self.assertEqual(result.lang2, lang2)
        self.assertAlmostEqual(result.raw_prob_1_to_2, 0.4206666946411133)
        self.assertAlmostEqual(result.raw_prob_2_to_1, 0.20401404798030853)
        self.assertIsNone(result.pvalue)

    def test_order_invariant(self):
        """detect(sentence1, sentence2) should be identical to detect(sentence2, sentence1)"""
        result1 = self.detector.detect(
            "Können Sie mir dabei weiter helfen?",
            "Pouvez-vous m'aider ?",
            "de",
            "fr",
        )
        result2 = self.detector.detect(
            "Pouvez-vous m'aider ?",
            "Können Sie mir dabei weiter helfen?",
            "fr",
            "de",
        )
        self.assertEqual(result1.predicted_direction, result2.predicted_direction)
        self.assertEqual(result1.lang1, result2.lang2)
        self.assertEqual(result1.lang2, result2.lang1)
        self.assertAlmostEqual(result1.raw_prob_1_to_2, result2.raw_prob_2_to_1)
        self.assertAlmostEqual(result1.raw_prob_2_to_1, result2.raw_prob_1_to_2)
        self.assertAlmostEqual(result1.prob_1_to_2, result2.prob_2_to_1)
        self.assertAlmostEqual(result1.prob_2_to_1, result2.prob_1_to_2)


class DocumentLevelTranslationDirectionDetectorTestCase(TestCase):

    def setUp(self):
        self.detector = TranslationDirectionDetector()

    def test_document(self):
        # Test repeated sentence – results should be identical to single-sentence
        sentences1 = 10 * ["Können Sie mir dabei weiter helfen?"]
        sentences2 = 10 * ["Pouvez-vous m'aider ?"]
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentences1, sentences2, lang1, lang2)
        self.assertEqual(result.predicted_direction, "de→fr")
        self.assertEqual(result.sentence1, sentences1)
        self.assertEqual(result.sentence2, sentences2)
        self.assertEqual(result.lang1, lang1)
        self.assertEqual(result.lang2, lang2)
        self.assertAlmostEqual(result.raw_prob_1_to_2, 0.4206666946411133)
        self.assertAlmostEqual(result.raw_prob_2_to_1, 0.20401404798030853)
        self.assertIsNone(result.pvalue)
        # Test two unique sentences
        sentences1 = ["Können Sie mir dabei weiter helfen?", "Mit freundlichen Grüßen"]
        sentences2 = ["Pouvez-vous m'aider ?", "Cordialement"]
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentences1, sentences2, lang1, lang2)
        self.assertEqual(result.predicted_direction, "de→fr")
        self.assertEqual(result.sentence1, sentences1)
        self.assertEqual(result.sentence2, sentences2)
        self.assertEqual(result.lang1, lang1)
        self.assertEqual(result.lang2, lang2)
        self.assertAlmostEqual(result.raw_prob_1_to_2, 0.24544709258493533)
        self.assertAlmostEqual(result.raw_prob_2_to_1, 0.23611102362602063)
        self.assertIsNone(result.pvalue)

    def test_single_sentence(self):
        # Test repeated sentence – results should be identical to single-sentence
        sentences1 = ["Können Sie mir dabei weiter helfen?"]
        sentences2 = ["Pouvez-vous m'aider ?"]
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentences1, sentences2, lang1, lang2)
        with self.assertRaises(ValueError):
            self.detector.detect(sentences1, sentences2, lang1, lang2, return_pvalue=True)

    def test_permutation_test(self):
        sentences1 = 10 * ["Können Sie mir dabei weiter helfen?"]
        sentences2 = 10 * ["Pouvez-vous m'aider ?"]
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentences1, sentences2, lang1, lang2, return_pvalue=True)
        self.assertEqual(result.predicted_direction, "de→fr")
        self.assertEqual(result.sentence1, sentences1)
        self.assertEqual(result.sentence2, sentences2)
        self.assertEqual(result.lang1, lang1)
        self.assertEqual(result.lang2, lang2)
        self.assertAlmostEqual(result.raw_prob_1_to_2, 0.4206666946411132)
        self.assertAlmostEqual(result.raw_prob_2_to_1, 0.2040140479803085)
        self.assertEqual(result.pvalue, 0.001953125)
        # Test two unique sentences
        sentences1 = ["Können Sie mir dabei weiter helfen?", "Mit freundlichen Grüßen"]
        sentences2 = ["Pouvez-vous m'aider ?", "Cordialement"]
        lang1 = "de"
        lang2 = "fr"
        result = self.detector.detect(sentences1, sentences2, lang1, lang2, return_pvalue=True)
        self.assertEqual(result.predicted_direction, "de→fr")
        self.assertEqual(result.sentence1, sentences1)
        self.assertEqual(result.sentence2, sentences2)
        self.assertEqual(result.lang1, lang1)
        self.assertEqual(result.lang2, lang2)
        self.assertAlmostEqual(result.raw_prob_1_to_2, 0.24544709258493533)
        self.assertAlmostEqual(result.raw_prob_2_to_1, 0.23611102362602063)
        self.assertEqual(result.pvalue, 1.0)

    def test_document_order_invariant(self):
        """detect(document1, document2) should be identical to detect(document2, document1)"""
        result1 = self.detector.detect(
            ["Können Sie mir dabei weiter helfen?", "Mit freundlichen Grüßen"],
            ["Pouvez-vous m'aider ?", "Cordialement"],
            "de",
            "fr",
            return_pvalue=True,
        )
        result2 = self.detector.detect(
            ["Pouvez-vous m'aider ?", "Cordialement"],
            ["Können Sie mir dabei weiter helfen?", "Mit freundlichen Grüßen"],
            "fr",
            "de",
            return_pvalue=True,
        )
        self.assertEqual(result1.predicted_direction, result2.predicted_direction)
        self.assertEqual(result1.lang1, result2.lang2)
        self.assertEqual(result1.lang2, result2.lang1)
        self.assertAlmostEqual(result1.raw_prob_1_to_2, result2.raw_prob_2_to_1)
        self.assertAlmostEqual(result1.raw_prob_2_to_1, result2.raw_prob_1_to_2)
        self.assertAlmostEqual(result1.prob_1_to_2, result2.prob_2_to_1)
        self.assertAlmostEqual(result1.prob_2_to_1, result2.prob_1_to_2)
        self.assertEqual(result1.pvalue, result2.pvalue)

    def test_sentence_order_invariant(self):
        """detect([sentence1, sentence2], ...) should be identical to detect([sentence2, sentence1], ...)"""
        result1 = self.detector.detect(
            ["Können Sie mir dabei weiter helfen?", "Mit freundlichen Grüßen"],
            ["Pouvez-vous m'aider ?", "Cordialement"],
            "de",
            "fr",
            return_pvalue=True,
        )
        result2 = self.detector.detect(
            ["Mit freundlichen Grüßen", "Können Sie mir dabei weiter helfen?"],
            ["Cordialement", "Pouvez-vous m'aider ?"],
            "de",
            "fr",
            return_pvalue=True,
        )
        self.assertEqual(result1.predicted_direction, result2.predicted_direction)
        self.assertEqual(result1.lang1, result2.lang1)
        self.assertEqual(result1.lang2, result2.lang2)
        self.assertAlmostEqual(result1.raw_prob_1_to_2, result2.raw_prob_1_to_2)
        self.assertAlmostEqual(result1.raw_prob_2_to_1, result2.raw_prob_2_to_1)
        self.assertAlmostEqual(result1.prob_1_to_2, result2.prob_1_to_2)
        self.assertAlmostEqual(result1.prob_2_to_1, result2.prob_2_to_1)
        self.assertEqual(result1.pvalue, result2.pvalue)

    def test_get_sentence_length(self):
        sentence = "This is a test."
        self.assertEqual(self.detector._get_sentence_length(sentence), 5)
