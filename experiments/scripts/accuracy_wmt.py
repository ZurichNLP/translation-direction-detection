import logging
from collections import defaultdict

from experiments.cached_model import CachedTranslationModel
from experiments.utils import evaluate_sentence_level

from nmtscore import NMTScorer
import numpy as np

from experiments.datasets import load_all_datasets
from translation_direction_detection.detector import TranslationDirectionDetector

logging.basicConfig(level=logging.INFO)

"""
\begin{tabularx}{\textwidth}{@{}Xrrrrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
\midrule
\midrule
en\biarrow cs & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow de & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow ru & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow uk & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow zh & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
cs\biarrow uk & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
de\biarrow fr & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\addlinespace
Macro-Avg. & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\bottomrule
\end{tabularx}
"""

LANG_PAIRS = ["en-cs", "en-de", "en-ru", "en-uk", "en-zh", "cs-uk", "de-fr"]
USE_NORMALIZATION = False

model_names = ["facebook/m2m100_418M", "alirezamsh/small100", "facebook/nllb-200-1.3B"]
detectors = []

for model_name in model_names:
    model = CachedTranslationModel(model_name)
    scorer = NMTScorer(model)
    detector = TranslationDirectionDetector(scorer, use_normalization=USE_NORMALIZATION)
    detectors.append(detector)

datasets = load_all_datasets()
"""data_subsets = [
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "nmt" and dataset.name != "wmt16"],
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "ht" and dataset.name != "wmt16"],
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "pre-nmt"],
]"""

wmt16_accuracies_dict = defaultdict(list)
wmt22_accuracies_dict = defaultdict(list)
wmt23_accuracies_dict = defaultdict(list)

for model_name, detector in zip(model_names, detectors):
    for lang_pair in LANG_PAIRS:
        lang1, lang2 = lang_pair.split("-")
        wmt16_datasets = [dataset for dataset in datasets if dataset.name == f"wmt16" and (dataset.translation_direction == f"{lang1}→{lang2}" or dataset.translation_direction==f"{lang2}→{lang1}") and dataset.type != 'pre-nmt']
        wmt16_accuracies = evaluate_sentence_level(detector, wmt16_datasets) if len(wmt16_datasets) > 0 else 0
        wmt16_accuracies_dict[model_name].append(wmt16_accuracies)
        wmt22_datasets = [dataset for dataset in datasets if dataset.name == f"wmt22" and (dataset.translation_direction == f"{lang1}→{lang2}" or dataset.translation_direction==f"{lang2}→{lang1}")]
        wmt22_accuracies = evaluate_sentence_level(detector, wmt22_datasets) if len(wmt22_datasets) > 0 else 0
        wmt22_accuracies_dict[model_name].append(wmt22_accuracies)
        wmt23_datasets = [dataset for dataset in datasets if dataset.name == f"wmt23" and (dataset.translation_direction == f"{lang1}→{lang2}" or dataset.translation_direction==f"{lang2}→{lang1}")]
        wmt23_accuracies = evaluate_sentence_level(detector, wmt23_datasets) if len(wmt23_datasets) > 0 else 0
        wmt23_accuracies_dict[model_name].append(wmt23_accuracies)
        for dataset in datasets:
            print(f"lp: {dataset.lang_pair}")
            print(f"lp: {dataset.name}")
            print(f"num examples: {dataset.num_examples}")

print(r"\begin{tabular}{cccccccccc}")
print(r"\toprule")
print(r"& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\")
print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
print(r"Language Pair & WMT16 & WMT22 & WMT23 & WMT16 & WMT21 & WMT22 & WMT16 & WMT22 & WMT23 \\")
print(r"\midrule")

for i, lang_pair in enumerate(LANG_PAIRS):
    print(lang_pair.replace("-", "\\biarrow ") + " & ", end="")
    for j, model_name in enumerate(model_names):
        print(f"{wmt16_accuracies_dict[model_name][i]:.2f} & ", end="")
        print(f"{wmt22_accuracies_dict[model_name][i]:.2f} & ", end="")
        print(f"{wmt23_accuracies_dict[model_name][i]:.2f} ", end="")
        if j < len(model_names) - 1:
            print("& ", end="")
    print(r"\\", end="")
    print()

print(r"\addlinespace")
print(r"Macro-Avg. & ", end="")
for i, model_name in enumerate(model_names):
    print(f"{np.mean([_ for _ in wmt16_accuracies_dict[model_name]if _ != 0.00]):.2f} & ", end="")
    print(f"{np.mean([_ for _ in wmt22_accuracies_dict[model_name]if _ != 0.00]):.2f} & ", end="")
    print(f"{np.mean([_ for _ in wmt23_accuracies_dict[model_name]if _ != 0.00]):.2f} ", end="")
    if i < len(model_names) - 1:
        print("& ", end="")
print(r"\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print()
print()