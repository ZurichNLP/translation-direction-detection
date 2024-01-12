from unittest import TestCase

from nmtscore import NMTScorer

from experiments.cached_model import CachedTranslationModel


class CachedModelTestCase(TestCase):

    def setUp(self):
        self.cached_model = CachedTranslationModel("alirezamsh_small100")
        self.model = NMTScorer().model

    def test_score(self):
        computed_score = self.model.score(
            tgt_lang="en",
            source_sentences="Sněmovna podpořila větší podíl státu z poplatků za uhlí.",
            hypothesis_sentences="MPs support greater state share in coal fees",
            src_lang="cs",
        )
        cached_score = self.cached_model.score(
            tgt_lang="en",
            source_sentences="Sněmovna podpořila větší podíl státu z poplatků za uhlí.",
            hypothesis_sentences="MPs support greater state share in coal fees",
            src_lang="cs",
        )
        self.assertAlmostEqual(computed_score, cached_score, places=3)
