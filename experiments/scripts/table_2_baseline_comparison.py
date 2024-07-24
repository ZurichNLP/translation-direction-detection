import sys
import pandas as pd
import os
import numpy as np
from pathlib import Path
import logging
from experiments.cached_model import CachedTranslationModel
from experiments.utils import evaluate_sentence_level

from nmtscore import NMTScorer
import numpy as np

from experiments.datasets import load_all_datasets
from translation_direction_detection.detector import TranslationDirectionDetector

logging.basicConfig(level=logging.INFO)

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "baseline_test_scores").absolute()) 

results = pd.DataFrame(columns=['source', 'target', 'type', 'predicted_label', 'gold_label'])

sup = {
    'encs': [],
    'enru': [],
    'ende': []
}

unsup = {
    'encs': [],
    'enru': [],
    'ende': []
}

# load results
LANG_PAIRS = ["en-cs", "en-de", "en-ru"]

for lang_pair in LANG_PAIRS:
    checkpoint_dir = "1e-05/checkpoint-1400" if 'de' in lang_pair else "1e-05/checkpoint-700"
    save_filename = "-".join(checkpoint_dir.split("/")[-2:])
    save_filename = save_filename.replace("checkpoints_", "" )
    lang_pair_split = lang_pair.split('-')
    lang_pair_rev = lang_pair_split[1]+"-"+lang_pair_split[0]
    for dir in os.listdir(os.environ["NMTSCORE_CACHE"]):
        if dir not in ['.empty', '__pycache__', 'junk.py']:
            result_path = os.path.join(os.environ["NMTSCORE_CACHE"], dir, f'{lang_pair_rev + "_" + save_filename}.csv')
            result_shard = pd.read_csv(result_path)
            results = pd.concat([results, result_shard], ignore_index=True)

# calculate accuracies
ht_accuracies = {
    'forward': [],
    'backward': [],
    'average': []
}

nmt_accuracies = {
    'forward': [],
    'backward': [],
    'average': []
}


for t in ['ht', 'nmt']:
    for lang_pair in LANG_PAIRS:
        lang1, lang2 = lang_pair.split("-")
        forward_results = results.loc[(results['gold_label'] == f"{lang1}→{lang2}") & (results['type'] == t)]
        if len(forward_results.index) > 0:
            forward_accuracy = (len(forward_results.loc[forward_results['gold_label'] == forward_results['predicted_label']].index)/len(forward_results.index))*100
        else:
            forward_accuracy = 0
        if t == 'ht':
            ht_accuracies['forward'].append(forward_accuracy)
        elif t == 'nmt':
            nmt_accuracies['forward'].append(forward_accuracy)
        backwards_result = results.loc[(results['gold_label'] == f"{lang2}→{lang1}") & (results['type'] == t)]
        if len(backwards_result.index) > 0:
            backward_accuracy = (len(backwards_result.loc[backwards_result['gold_label'] == backwards_result['predicted_label']].index)/len(backwards_result.index))*100
        else:
            backward_accuracy = 0
        if t == 'ht':
            ht_accuracies['backward'].append(backward_accuracy)
        elif t == 'nmt':
            nmt_accuracies['backward'].append(backward_accuracy)
        avg_accuracy = (forward_accuracy + backward_accuracy) / 2
        if t == 'ht':
            ht_accuracies['average'].append(avg_accuracy)
        elif t == 'nmt':
            nmt_accuracies['average'].append(avg_accuracy)
        sup[lang_pair.replace("-", "")].append(np.mean(avg_accuracy))
    print()

# unsupervised
USE_NORMALIZATION = False

# model_name = "alirezamsh/small100"
model_name = "facebook/m2m100_418M"
# model_name = "facebook/nllb-200-1.3B"
model = CachedTranslationModel(model_name)
scorer = NMTScorer(model)
detector = TranslationDirectionDetector(scorer, use_normalization=USE_NORMALIZATION)

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if not dataset.is_indirect and (dataset.type == "nmt" or dataset.type == "ht") and dataset.name != "wmt16"]

forward_accuracies = []
backward_accuracies = []
avg_accuracies = []

for lang_pair in LANG_PAIRS:
    lang1, lang2 = lang_pair.split("-")
    forward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang1}→{lang2}"]
    forward_accuracy = evaluate_sentence_level(detector, forward_datasets)
    forward_accuracies.append(forward_accuracy)
    backward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang2}→{lang1}"]
    backward_accuracy = evaluate_sentence_level(detector, backward_datasets)
    backward_accuracies.append(backward_accuracy)
    avg_accuracy = (forward_accuracy + backward_accuracy) / 2
    unsup[lang_pair.replace("-", "")].append(avg_accuracy)
print()

# format into table
r"""
\begin{tabular}{cccccccc}
\toprule
 & en\biarrow cs & en\biarrow ru & en\biarrow de\\
\midrule
Supervised & tbs & tba & tba \\
\mbox{Unsupervised (ours)} & tba & tba & tba \\
\bottomrule
\end{tabular}
\end{tabular}
"""

print(r'\begin{tabular}{cccc}')
print(r'\toprule')
print(r' & en\biarrow cs & en\biarrow ru & en\biarrow de\\' )
print(r'\midrule')
print("Supervised" + " & ", end="")
print(f"{np.mean(sup['encs']):.2f} & ", end="")
print(f"{np.mean(sup['enru']):.2f} & ", end="")
print(f"{np.mean(sup['ende']):.2f} \\\\", end="")
print()
print(r"\mbox{Unsupervised (ours)}" + " & ", end="")
print(f"{np.mean(unsup['encs']):.2f} & ", end="")
print(f"{np.mean(unsup['enru']):.2f} & ", end="")
print(f"{np.mean(unsup['ende']):.2f} \\\\", end="")
print()
print(r"\bottomrule")
print(r"\end{tabular}")  

