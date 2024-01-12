from unittest import TestCase

from experiments.datasets import load_wmt16_dataset, load_wmt21_23_dataset, load_all_datasets


class WMT16TestCase(TestCase):

    def setUp(self):
        self.ht_dataset = load_wmt16_dataset("cs-en", "ht")
        self.nmt_dataset = load_wmt16_dataset("cs-en", "nmt")
        self.pre_nmt_dataset = load_wmt16_dataset("cs-en", "pre-nmt")

    def test_num_examples(self):
        self.assertEqual(len(self.ht_dataset.examples), self.ht_dataset.num_systems * self.ht_dataset.num_sentences)
        self.assertEqual(len(self.nmt_dataset.examples), self.nmt_dataset.num_systems * self.nmt_dataset.num_sentences)
        self.assertEqual(len(self.pre_nmt_dataset.examples), self.pre_nmt_dataset.num_systems * self.pre_nmt_dataset.num_sentences)

    def test_num_sentences(self):
        self.assertEqual(self.ht_dataset.num_sentences, 1499)
        self.assertEqual(self.nmt_dataset.num_sentences, 1499)
        self.assertEqual(self.pre_nmt_dataset.num_sentences, 1499)

    def test_num_documents(self):
        self.assertEqual(self.ht_dataset.num_documents, 48)
        self.assertEqual(self.nmt_dataset.num_documents, 48)
        self.assertEqual(self.pre_nmt_dataset.num_documents, 48)

    def test_num_systems(self):
        self.assertEqual(self.ht_dataset.num_systems, 1)
        self.assertEqual(self.nmt_dataset.num_systems, 1)
        self.assertEqual(self.pre_nmt_dataset.num_systems, 11)

    def test_num_system_documents(self):
        self.assertEqual(self.ht_dataset.num_system_documents, 1 * 48)
        self.assertEqual(self.nmt_dataset.num_system_documents, 1 * 48)
        self.assertEqual(self.pre_nmt_dataset.num_system_documents, 11 * 48)

    def test_translation_direction(self):
        self.assertEqual(self.ht_dataset.translation_direction, "cs→en")
        self.assertEqual(self.nmt_dataset.translation_direction, "cs→en")
        self.assertEqual(self.pre_nmt_dataset.translation_direction, "cs→en")

    def test_lang_pair(self):
        self.assertEqual(self.ht_dataset.lang_pair, "cs↔en")
        self.assertEqual(self.nmt_dataset.lang_pair, "cs↔en")
        self.assertEqual(self.pre_nmt_dataset.lang_pair, "cs↔en")

    def test_documents(self):
        for dataset in [self.ht_dataset, self.nmt_dataset, self.pre_nmt_dataset]:
            self.assertEqual(len(dataset.documents), dataset.num_documents)
            for document in dataset.documents.values():
                num_systems = len(set([e.sysid for e in document]))
                self.assertEqual(num_systems, dataset.num_systems)

    def test_ht_example(self):
        example = self.ht_dataset.examples[0]
        self.assertEqual(example.src, "Sněmovna podpořila větší podíl státu z poplatků za uhlí.")
        self.assertEqual(example.tgt, "MPs support greater state share in coal fees")
        self.assertEqual(example.docid, "aktualne.cz.46941")
        self.assertEqual(example.sysid, "wmt16.cs-en.ref")

    def test_nmt_example(self):
        example = self.nmt_dataset.examples[0]
        self.assertEqual(example.src, "Sněmovna podpořila větší podíl státu z poplatků za uhlí.")
        self.assertEqual(example.tgt, "The House has supported a greater share of the state from the levy on coal.")
        self.assertEqual(example.docid, "aktualne.cz.46941")
        self.assertEqual(example.sysid, "newstest2016.uedin-nmt.4361.cs-en")


class WMT21TestCase(TestCase):

    def setUp(self):
        self.ht_dataset = load_wmt21_23_dataset("wmt21", "cs-en", "ht")
        self.nmt_dataset = load_wmt21_23_dataset("wmt21", "cs-en", "nmt")

    def test_num_sentences(self):
        self.assertEqual(self.ht_dataset.num_sentences, 1000)
        self.assertEqual(self.nmt_dataset.num_sentences, 1000)

    def test_num_documents(self):
        self.assertEqual(self.ht_dataset.num_documents, 62)
        self.assertEqual(self.nmt_dataset.num_documents, 62)

    def test_num_systems(self):
        self.assertEqual(self.ht_dataset.num_systems, 2)
        self.assertEqual(self.nmt_dataset.num_systems, 8)

    def test_num_system_documents(self):
        self.assertEqual(self.ht_dataset.num_system_documents, 2 * 62)
        self.assertEqual(self.nmt_dataset.num_system_documents, 8 * 62)

    def test_translation_direction(self):
        self.assertEqual(self.ht_dataset.translation_direction, "cs→en")
        self.assertEqual(self.nmt_dataset.translation_direction, "cs→en")

    def test_lang_pair(self):
        self.assertEqual(self.ht_dataset.lang_pair, "cs↔en")
        self.assertEqual(self.nmt_dataset.lang_pair, "cs↔en")

    def test_documents(self):
        for dataset in [self.ht_dataset, self.nmt_dataset]:
            self.assertEqual(len(dataset.documents), dataset.num_documents)
            for document in dataset.documents.values():
                num_systems = len(set([e.sysid for e in document]))
                self.assertEqual(num_systems, dataset.num_systems)

    def test_ht_example(self):
        example = self.ht_dataset.examples[0]
        self.assertEqual(example.src, "Novák pomohl Trabzonsporu k druhému místu, trefil se i Škoda")
        self.assertEqual(example.tgt, "Novák helped Trabzonspor to finish in second place, Škoda scored too")
        self.assertEqual(example.docid, "halo_noviny-cs-01.3089")
        self.assertEqual(example.sysid, "wmt21.cs-en.ref-A")

    def test_nmt_example(self):
        example = self.nmt_dataset.examples[0]
        self.assertEqual(example.src, "Novák pomohl Trabzonsporu k druhému místu, trefil se i Škoda")
        self.assertEqual(example.tgt, "Novák helped Trabzonspor to second place, Škoda also scored")
        self.assertEqual(example.docid, "halo_noviny-cs-01.3089")
        self.assertEqual(example.sysid, "newstest2021.cs-en.hyp.CUNI-DocTransformer.en")


class AllDatasetsTestCase(TestCase):

    def setUp(self):
        self.datasets = load_all_datasets()

    def test_examples(self):
        for dataset in self.datasets:
            if not dataset.examples:
                print(dataset)
            self.assertGreater(len(dataset.examples), 0)
