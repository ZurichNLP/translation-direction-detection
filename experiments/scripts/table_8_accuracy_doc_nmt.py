import logging

from experiments.cached_model import CachedTranslationModel
from experiments.utils import evaluate_document_level

from nmtscore import NMTScorer
import numpy as np

from experiments.datasets import load_all_datasets
from translation_direction_detection.detector import TranslationDirectionDetector

logging.basicConfig(level=logging.INFO)

LANG_PAIRS = ["en-cs", "en-de", "en-ru", "en-zh"]
USE_NORMALIZATION = False

# model_name = "alirezamsh/small100"
model_name = "facebook/m2m100_418M"
# model_name = "facebook/nllb-200-1.3B"
model = CachedTranslationModel(model_name)
scorer = NMTScorer(model)
detector = TranslationDirectionDetector(scorer, use_normalization=USE_NORMALIZATION)

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "nmt" and dataset.name != "wmt16"]

forward_accuracies = []
backward_accuracies = []
avg_accuracies = []

for lang_pair in LANG_PAIRS:
    lang1, lang2 = lang_pair.split("-")
    forward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang1}→{lang2}"]
    backward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang2}→{lang1}"]
    forward_accuracy = evaluate_document_level(detector, forward_datasets)
    forward_accuracies.append(forward_accuracy)
    backward_accuracy = evaluate_document_level(detector, backward_datasets)
    backward_accuracies.append(backward_accuracy)
    avg_accuracy = (forward_accuracy + backward_accuracy) / 2
    avg_accuracies.append(avg_accuracy)

print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Language Pair &  \(\rightarrow\) &  \(\leftarrow\) & Avg. \\")
print(r"\midrule")

for i, lang_pair in enumerate(LANG_PAIRS):
    print(lang_pair.replace("-", "\\biarrow ") + " & ", end="")
    print(f"{forward_accuracies[i]:.2f} & ", end="")
    print(f"{backward_accuracies[i]:.2f} & ", end="")
    print(f"{avg_accuracies[i]:.2f} \\\\", end="")
    print()

print(r"\addlinespace")
print(r"Macro-Avg. & ", end="")
print(f"{np.mean(forward_accuracies):.2f} & ", end="")
print(f"{np.mean(backward_accuracies):.2f} & ", end="")
print(f"{np.mean(avg_accuracies):.2f} \\\\", end="")
print()
print(r"\bottomrule")
print(r"\end{tabular}")
