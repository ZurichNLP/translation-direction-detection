import sys
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from pathlib import Path

method = sys.argv[1] 
checkpoint_dir = sys.argv[2]

'''
siamese_ht-nmt/checkpoint-1815
'''

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "cached_scores").absolute())

results = pd.DataFrame(columns=['source', 'target', 'type', 'predicted_label', 'gold_label'])

# load results
for dir in os.listdir(os.environ["NMTSCORE_CACHE"]):
    if dir not in ['.empty', '__pycache__', 'junk.py']:
        result_path = os.path.join(os.environ["NMTSCORE_CACHE"], dir, f'baseline-pooler_{"_".join(checkpoint_dir.split("/"))}.csv')
        result_shard = pd.read_csv(result_path)
        results = pd.concat([results, result_shard], ignore_index=True)

LANG_PAIRS = ["cs-en", "de-en", "ru-en"]

# calculate accuraciess
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

pre_nmt_accuracies = {
    'forward': [],
    'backward': [],
    'average': []
}

for t in ['ht', 'nmt', 'pre-nmt']:
    for lang_pair in LANG_PAIRS:
        print(lang_pair, t)
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
        elif t == 'pre-nmt':
            pre_nmt_accuracies['forward'].append(forward_accuracy)
        backwards_result = results.loc[(results['gold_label'] == f"{lang2}→{lang1}") & (results['type'] == t)]
        if len(backwards_result.index) > 0:
            backward_accuracy = (len(backwards_result.loc[backwards_result['gold_label'] == backwards_result['predicted_label']].index)/len(backwards_result.index))*100
        else:
            backward_accuracy = 0
        if t == 'ht':
            ht_accuracies['backward'].append(backward_accuracy)
        elif t == 'nmt':
            nmt_accuracies['backward'].append(backward_accuracy)
        elif t == 'pre-nmt':
            pre_nmt_accuracies['backward'].append(backward_accuracy)
        avg_accuracy = (forward_accuracy + backward_accuracy) / 2
        if t == 'ht':
            ht_accuracies['average'].append(avg_accuracy)
        elif t == 'nmt':
            nmt_accuracies['average'].append(avg_accuracy)
        elif t == 'pre-nmt':
            pre_nmt_accuracies['average'].append(avg_accuracy)
    print()

# format into table
r"""
\begin{tabular}{ccccccc}
\toprule
& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT} & \multicolumn{3}{c}{Pre-NMT} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-11}
 Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
\midrule
en\(\leftrightarrow\)cs & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\(\leftrightarrow\)de & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\(\leftrightarrow\)ru & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\addlinespace
Macro-Avg. & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\bottomrule
\end{tabular}
\end{tabular}
"""
print(r'\begin{tabular}{cccccccccc}')
print(r'\toprule')
print(r'& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT} & \multicolumn{3}{c}{Pre-NMT}  \\' )
print(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}')
print(r'Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg.  & \(\rightarrow\) & \(\leftarrow\) & Avg. \\')
print(r'\midrule')

for i, lang_pair in enumerate(LANG_PAIRS):
    lp = lang_pair.split('-')
    print(lp[1]+r"\biarrow "+lp[0] + " & ", end="")
    print(f"{ht_accuracies['backward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies['forward'][i]:.2f} & ", end="")
    print(f"{ht_accuracies['average'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies['backward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies['forward'][i]:.2f} & ", end="")
    print(f"{nmt_accuracies['average'][i]:.2f} & ", end="")
    print(f"{pre_nmt_accuracies['backward'][i]:.2f} & ", end="")
    print(f"{pre_nmt_accuracies['forward'][i]:.2f} & ", end="")
    print(f"{pre_nmt_accuracies['average'][i]:.2f} \\\\", end="")
    print()

print(r"\addlinespace")
print(r"Macro-Avg. & ", end="")
print(f"{np.mean(ht_accuracies['backward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies['forward']):.2f} & ", end="")
print(f"{np.mean(ht_accuracies['average']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies['backward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies['forward']):.2f} & ", end="")
print(f"{np.mean(nmt_accuracies['average']):.2f} & ", end="")
print(f"{np.mean(pre_nmt_accuracies['backward']):.2f} & ", end="")
print(f"{np.mean(pre_nmt_accuracies['forward']):.2f} & ", end="")
print(f"{np.mean(pre_nmt_accuracies['average']):.2f} \\\\", end="")
print()
print(r"\bottomrule")
print(r"\end{tabular}")       
