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

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "supervised_baseline" / "wmt"/ "baseline_test_scores").absolute()) 

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
LANG_PAIRS = ["en-cs", "en-ru", "en-de"]

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

ht_accuracies_unsup = {
    'forward': [],
    'backward': [],
    'average': []
}

nmt_accuracies_unsup = {
    'forward': [],
    'backward': [],
    'average': []
}

for t in ['ht', 'nmt']:
    for lang_pair in LANG_PAIRS:
        lang1, lang2 = lang_pair.split("-")
        forward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang1}→{lang2}" and dataset.type == t]
        forward_accuracy = evaluate_sentence_level(detector, forward_datasets)
        ht_accuracies_unsup['forward'].append(forward_accuracy) if t == 'ht' else nmt_accuracies_unsup['forward'].append(forward_accuracy)
        backward_datasets = [dataset for dataset in datasets if dataset.translation_direction == f"{lang2}→{lang1}" and dataset.type == t]
        backward_accuracy = evaluate_sentence_level(detector, backward_datasets)
        ht_accuracies_unsup['backward'].append(backward_accuracy) if t == 'ht' else nmt_accuracies_unsup['backward'].append(backward_accuracy)
        avg_accuracy = (forward_accuracy + backward_accuracy) / 2
        ht_accuracies_unsup['average'].append(avg_accuracy) if t == 'ht' else nmt_accuracies_unsup['average'].append(avg_accuracy)
print()

# format into table
r"""
\begin{tabular}{cccccccc}
\toprule
r"& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT} \\"
r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}"
r"LP & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\"
\midrule
en\biarrow cs & tbs & tba & tba & tbs & tba & tba \\
en\biarrow ru & tba & tba & tba & tbs & tba & tba \\
en\biarrow de & tba & tba & tba & tbs & tba & tba \\
\bottomrule
\end{tabular}
"""

print(r'\begin{tabular}{cccccccc}')
print(r'\toprule')
print(r"& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT} \\")
print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
print(r"LP & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & Avg.\\")
print(r'\midrule')
for i, lp in enumerate(LANG_PAIRS):
    lp = lp.split('-')
    print(lp[0]+r"\biarrow "+lp[1]+" (sup)" + " & ", end="")
    print(f"{ht_accuracies['forward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies['backward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies['average'][i]:.2f} &  ", end="")
    print(f"{nmt_accuracies['forward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies['backward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies['average'][i]:.2f} & ", end="")
    print(f"{np.mean([np.mean(ht_accuracies['average'][i]), np.mean(nmt_accuracies['average'][i])]):.2f} \\\\  ", end="")
    print()
    print(lp[0]+r"\biarrow "+lp[1]+" (unsup)" + " & ", end="")
    print(f"{ht_accuracies_unsup['forward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies_unsup['backward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies_unsup['average'][i]:.2f} &  ", end="")
    print(f"{nmt_accuracies_unsup['forward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies_unsup['backward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies_unsup['average'][i]:.2f} & ", end="")
    print(f"{np.mean([np.mean(ht_accuracies_unsup['average'][i]), np.mean(nmt_accuracies_unsup['average'][i])]):.2f} \\\\  ", end="")
    print()
print(r"\addlinespace")
print(r"Macro-Avg. & ", end="")
print(f"{np.mean(ht_accuracies['forward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies['backward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies['average']):.2f} &  ", end="")
print(f"{np.mean(nmt_accuracies['forward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies['backward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies['average']):.2f} & ", end="")
print(f"{np.mean([np.mean(nmt_accuracies['average']), np.mean(ht_accuracies['average'])]):.2f} \\\\ ", end="")
print()
print(r"Macro-Avg. & ", end="")
print(f"{np.mean(ht_accuracies_unsup['forward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies_unsup['backward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies_unsup['average']):.2f} &  ", end="")
print(f"{np.mean(nmt_accuracies_unsup['forward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies_unsup['backward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies_unsup['average']):.2f} & ", end="")
print(f"{np.mean([np.mean(nmt_accuracies_unsup['average']), np.mean(ht_accuracies_unsup['average'])]):.2f} \\\\ ", end="")
print()
print(r"\bottomrule")
print(r"\end{tabular}")     