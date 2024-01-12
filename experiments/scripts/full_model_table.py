import logging
from collections import defaultdict

from experiments.cached_model import CachedTranslationModel
from experiments.utils import evaluate_sentence_level, evaluate_document_level

from nmtscore import NMTScorer
import numpy as np

from experiments.datasets import load_all_datasets
from translation_direction_detection.detector import TranslationDirectionDetector

logging.basicConfig(level=logging.INFO)

r"""
\begin{tabular}{cccccccccc}
\toprule
& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
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
\end{tabular}

\begin{tabular}{cccccccccc}
\toprule
& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
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
\end{tabular}

\toprule
& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
\midrule
en\biarrow cs & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow de & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\biarrow ru & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\addlinespace
Macro-Avg. & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\bottomrule
\end{tabular}
"""

LANG_PAIRS_NMT = ["en-cs", "en-de", "en-ru", "en-uk", "en-zh", "cs-uk", "de-fr"]
LANG_PAIRS_HT = ["en-cs", "en-de", "en-ru", "en-uk", "en-zh", "cs-uk", "de-fr"]
LANG_PAIRS_PRENMT = ["en-cs", "en-de", "en-ru"]
ALL_LANG_PAIRS = [LANG_PAIRS_NMT, LANG_PAIRS_HT, LANG_PAIRS_PRENMT]
LANG_PAIRS_DOC_NMT = ["en-cs", "en-de", "en-ru", "en-zh"]
LANG_PAIRS_DOC_HT = ["en-cs", "en-de", "en-ru", "en-zh"]
USE_NORMALIZATION = False

model_names = ["facebook/m2m100_418M", "alirezamsh/small100", "facebook/nllb-200-1.3B"]
detectors = []
for model_name in model_names:
    model = CachedTranslationModel(model_name)
    scorer = NMTScorer(model)
    detector = TranslationDirectionDetector(scorer, use_normalization=USE_NORMALIZATION)
    detectors.append(detector)

datasets = load_all_datasets()
data_subsets = [
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "nmt" and dataset.name != "wmt16"],
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "ht" and dataset.name != "wmt16"],
    [dataset for dataset in datasets if not dataset.is_indirect and dataset.type == "pre-nmt"],
]

# Sentence level
for data_subset, lang_pairs in zip(data_subsets, ALL_LANG_PAIRS):
    model_forward_accuracies = defaultdict(list)
    model_backward_accuracies = defaultdict(list)
    model_avg_accuracies = defaultdict(list)

    for model_name, detector in zip(model_names, detectors):
        for lang_pair in lang_pairs:
            lang1, lang2 = lang_pair.split("-")
            forward_datasets = [dataset for dataset in data_subset if dataset.translation_direction == f"{lang1}→{lang2}"]
            forward_accuracy = evaluate_sentence_level(detector, forward_datasets)
            model_forward_accuracies[model_name].append(forward_accuracy)
            backward_datasets = [dataset for dataset in data_subset if dataset.translation_direction == f"{lang2}→{lang1}"]
            backward_accuracy = evaluate_sentence_level(detector, backward_datasets)
            model_backward_accuracies[model_name].append(backward_accuracy)
            avg_accuracy = (forward_accuracy + backward_accuracy) / 2
            model_avg_accuracies[model_name].append(avg_accuracy)

    print(r"\begin{tabular}{cccccccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    print(r"Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\")
    print(r"\midrule")

    for i, lang_pair in enumerate(lang_pairs):
        print(lang_pair.replace("-", "\\biarrow ") + " & ", end="")
        for j, model_name in enumerate(model_names):
            print(f"{model_forward_accuracies[model_name][i]:.2f} & ", end="")
            print(f"{model_backward_accuracies[model_name][i]:.2f} & ", end="")
            is_best_model = f"{model_avg_accuracies[model_name][i]:.2f}" == f"{max([model_avg_accuracies[model_name][i] for model_name in model_names]):.2f}"
            if is_best_model:
                print(f"\\textbf{{{model_avg_accuracies[model_name][i]:.2f}}} ", end="")
            else:
                print(f"{model_avg_accuracies[model_name][i]:.2f} ", end="")
            if j < len(model_names) - 1:
                print("& ", end="")
        print(r"\\", end="")
        print()

    print(r"\addlinespace")
    print(r"Macro-Avg. & ", end="")
    for i, model_name in enumerate(model_names):
        print(f"{np.mean(model_forward_accuracies[model_name]):.2f} & ", end="")
        print(f"{np.mean(model_backward_accuracies[model_name]):.2f} & ", end="")
        is_best_model = f"{np.mean(model_avg_accuracies[model_name]):.2f}" == f"{max([np.mean(model_avg_accuracies[model_name]) for model_name in model_names]):.2f}"
        if is_best_model:
            print(f"\\textbf{{{np.mean(model_avg_accuracies[model_name]):.2f}}} ", end="")
        else:
            print(f"{np.mean(model_avg_accuracies[model_name]):.2f} ", end="")
        if i < len(model_names) - 1:
            print("& ", end="")
    print(r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print()

# Document level
for data_subset, lang_pairs in zip(data_subsets[:2], [LANG_PAIRS_DOC_NMT, LANG_PAIRS_DOC_HT]):
    model_forward_accuracies = defaultdict(list)
    model_backward_accuracies = defaultdict(list)
    model_avg_accuracies = defaultdict(list)

    for lang_pair in lang_pairs:
        lang1, lang2 = lang_pair.split("-")
        forward_datasets = [dataset for dataset in data_subset if dataset.translation_direction == f"{lang1}→{lang2}"]
        backward_datasets = [dataset for dataset in data_subset if dataset.translation_direction == f"{lang2}→{lang1}"]
        for model_name, detector in zip(model_names, detectors):
            forward_accuracy = evaluate_document_level(detector, forward_datasets)
            model_forward_accuracies[model_name].append(forward_accuracy)
            backward_accuracy = evaluate_document_level(detector, backward_datasets)
            model_backward_accuracies[model_name].append(backward_accuracy)
            avg_accuracy = (forward_accuracy + backward_accuracy) / 2
            model_avg_accuracies[model_name].append(avg_accuracy)

    print(r"\begin{tabular}{cccccccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c}{M2M-100-418M} & \multicolumn{3}{c}{SMaLL-100} & \multicolumn{3}{c}{NLLB-200-1.3B} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    print(r"Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\")
    print(r"\midrule")

    for i, lang_pair in enumerate(lang_pairs):
        print(lang_pair.replace("-", "\\biarrow ") + " & ", end="")
        for j, model_name in enumerate(model_names):
            print(f"{model_forward_accuracies[model_name][i]:.2f} & ", end="")
            print(f"{model_backward_accuracies[model_name][i]:.2f} & ", end="")
            is_best_model = f"{model_avg_accuracies[model_name][i]:.2f}" == f"{max([model_avg_accuracies[model_name][i] for model_name in model_names]):.2f}"
            if is_best_model:
                print(f"\\textbf{{{model_avg_accuracies[model_name][i]:.2f}}} ", end="")
            else:
                print(f"{model_avg_accuracies[model_name][i]:.2f} ", end="")
            if j < len(model_names) - 1:
                print("& ", end="")
        print(r"\\", end="")
        print()

    print(r"\addlinespace")
    print(r"Macro-Avg. & ", end="")
    for i, model_name in enumerate(model_names):
        print(f"{np.mean(model_forward_accuracies[model_name]):.2f} & ", end="")
        print(f"{np.mean(model_backward_accuracies[model_name]):.2f} & ", end="")
        is_best_model = f"{np.mean(model_avg_accuracies[model_name]):.2f}" == f"{max([np.mean(model_avg_accuracies[model_name]) for model_name in model_names]):.2f}"
        if is_best_model:
            print(f"\\textbf{{{np.mean(model_avg_accuracies[model_name]):.2f}}} ", end="")
        else:
            print(f"{np.mean(model_avg_accuracies[model_name]):.2f} ", end="")
        if i < len(model_names) - 1:
            print("& ", end="")
    print(r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print()
