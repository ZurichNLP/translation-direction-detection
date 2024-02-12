import logging

from experiments.cached_model import CachedTranslationModel
from experiments.utils import evaluate_sentence_level

from nmtscore import NMTScorer
import numpy as np

from experiments.datasets import load_all_datasets
from translation_direction_detection.detector import TranslationDirectionDetector

logging.basicConfig(level=logging.INFO)

LANG_PAIRS = ["de-fr", "cs-uk", "bn-hi", "xh-zu"]
USE_NORMALIZATION = False

# model_name = "alirezamsh/small100"
model_name = "facebook/m2m100_418M"
# model_name = "facebook/nllb-200-1.3B"
model = CachedTranslationModel(model_name)
scorer = NMTScorer(model)
detector = TranslationDirectionDetector(scorer, use_normalization=USE_NORMALIZATION)

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if dataset.is_indirect and dataset.type == "ht"]

forward_accuracies = []
backward_accuracies = []
avg_accuracies = []

for lang_pair in LANG_PAIRS:
    lang1, lang2 = lang_pair.split("-")
    forward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang1}→{lang2}"]
    forward_accuracy = evaluate_sentence_level(detector, forward_datasets)
    forward_accuracies.append(forward_accuracy)
    backward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang2}→{lang1}"]
    backward_accuracy = 100-forward_accuracy 
    backward_accuracies.append(backward_accuracy)
print()

print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Language Pair &  \(\rightarrow\) &  \(\leftarrow\)\\")
print(r"\midrule")

for i, lang_pair in enumerate(LANG_PAIRS):
    print(lang_pair.replace("-", "\\biarrow ") + " & ", end="")
    print(f"{forward_accuracies[i]:.2f} & ", end="")
    print(f"{backward_accuracies[i]:.2f} \\\\ ", end="")
    print()


print(r"\bottomrule")
print(r"\end{tabular}")
