import csv
from pathlib import Path
from typing import List

from nmtscore import NMTScorer

from experiments.datasets import TranslationExample
from translation_direction_detection import TranslationDirectionDetector

csv_path = Path(__file__).parent.parent / 'data' / 'colchicine' / 'colchicine.csv'
assert csv_path.exists()


document: List[TranslationExample] = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        assert row['lang_pair'] == 'deen'
        assert row['src_lang'] == 'de'
        assert row['tgt_lang'] == 'en'
        example = TranslationExample(
            src=row['source'],
            tgt=row['eng_translation'],
            docid=None,
            sysid=None,
        )
        document.append(example)

print(f"Number of sentences: {len(document)}")
print(f"Example: {document[0]}")

scorer = NMTScorer("m2m100_418M")
detector = TranslationDirectionDetector(scorer)

result = detector.detect(
    sentence1=[example.src for example in document],
    sentence2=[example.tgt for example in document],
    lang1='de',
    lang2='en',
    return_pvalue=True,
    pvalue_n_resamples=10_000,
)
print(result)
