import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from experiments.utils import nllb_language_codes

shard = sys.argv[1]
assert int(shard) in range(12)

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "cached_scores" / f"nmtscore_cache{shard}").absolute())

from nmtscore import NMTScorer

from translation_direction_detection.detector import TranslationDirectionDetector
from experiments.datasets import load_all_datasets

logging.basicConfig(level=logging.INFO)

datasets = load_all_datasets()

scorer = NMTScorer("nllb-200-1.3B", device=0)
detector = TranslationDirectionDetector(scorer, use_normalization=True)

for i, dataset in enumerate(datasets):
    if i % 12 != int(shard):
        continue
    print(dataset)
    for example in tqdm(dataset.examples):
        detector.detect(
            sentence1=example.src,
            sentence2=example.tgt,
            lang1=nllb_language_codes[dataset.src_lang],
            lang2=nllb_language_codes[dataset.tgt_lang],
            score_kwargs={"use_cache": True},
        )
