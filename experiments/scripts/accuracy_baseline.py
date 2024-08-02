import sys
import pandas as pd
import os
import numpy as np
from pathlib import Path

split = sys.argv[1]
assert split in ['test', 'val']
checkpoint_dir = sys.argv[2]
train_set = sys.argv[4] 
assert train_set in ['wmt', 'europarl']
test_set = sys.argv[5] 
assert test_set in ['wmt', 'europarl']

'''
1e-05/checkpoint-[700, 1400, 1750]
'''

if train_set == 'wmt' and test_set == 'wmt':
    os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "supervised_baseline" / "wmt_baseline_validation_scores" / f"scores{shard}").absolute()) if split == 'val' else str((Path(__file__).parent.parent / "supervised_baseline" / "wmt_baseline_test_scores" / f"scores{shard}").absolute()) 
elif train_set == "europarl" and test_set == 'wmt':
    os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "supervised_baseline_europarl" / "europarl_wmt_baseline_validation_scores" / f"scores{shard}").absolute()) if split == 'val' else str((Path(__file__).parent.parent / "supervised_baseline_europarl" / "europarl_wmt_baseline_test_scores" / f"scores{shard}").absolute()) 
elif train_set == "europarl" and test_set == 'europarl':
    os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "supervised_baseline_europarl"/ "europarl_baseline_validation_scores" / f"scores{shard}").absolute()) if split == 'val' else str((Path(__file__).parent.parent / "supervised_baseline_europarl" / "europarl_baseline_test_scores" / f"scores{shard}").absolute()) 

results = pd.DataFrame(columns=['source', 'target', 'type', 'predicted_label', 'gold_label'])

# load results
LANG_PAIRS = ["en-cs", "en-de", "en-ru", "de-fr"]
save_filename = "-".join(checkpoint_dir.split("/")[-2:])
save_filename = save_filename.replace("checkpoints_", "" )

for lang_pair in LANG_PAIRS:
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

pre_nmt_accuracies = {
    'forward': [],
    'backward': [],
    'average': []
}

print("\n"+checkpoint_dir+":")
for t in ['ht', 'nmt', 'pre-nmt']:
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
\begin{tabular}{cccccccc}
\toprule
& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT} & \multicolumn{3}{c}{Pre-NMT} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-11}
 Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. &  LP Avg.\\
\midrule
en\(\leftrightarrow\)cs & tba & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\(\leftrightarrow\)de & tba & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
en\(\leftrightarrow\)ru & tba & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\addlinespace
Macro-Avg. & tba & tba & tba & tba & tba & tba & tba & tba & tba & tba \\
\bottomrule
\end{tabular}
\end{tabular}
"""

if split == 'val':
    print(r'\begin{tabular}{ccccccccccc}')
    print(r'\toprule')
    print(r'& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT}& \multicolumn{3}{c}{Pre-NMT}  \\' )
    print(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}')
    print(r'Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. &  LP Avg.\\')
    print(r'\midrule')

    for i, lang_pair in enumerate(LANG_PAIRS):
        lp = lang_pair.split('-')
        print(lp[0]+r"\biarrow "+lp[1] + " & ", end="")
        print(f"{ht_accuracies['forward'][i]:.2f} & ", end="")
        print(f"{ht_accuracies['backward'][i]:.2f} & ", end="")
        print(f"{ht_accuracies['average'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['forward'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['backward'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['average'][i]:.2f} & ", end="")
        print(f"{pre_nmt_accuracies['backward'][i]:.2f} & ", end="")
        print(f"{pre_nmt_accuracies['forward'][i]:.2f} & ", end="")
        print(f"{pre_nmt_accuracies['average'][i]:.2f} & ", end="")
        print(f"{np.mean([ht_accuracies['average'][i], pre_nmt_accuracies['average'][i], nmt_accuracies['average'][i]]):.2f} \\\\ ", end="")
        print()

    print(r"\addlinespace")
    print(r"Macro-Avg. & ", end="")
    print(f"{np.mean(ht_accuracies['forward']):.2f} & ", end="")
    print(f"{np.mean(ht_accuracies['backward']):.2f} & ", end="")
    print(f"{np.mean(ht_accuracies['average']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['forward']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['backward']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['average']):.2f} & ", end="")
    print(f"{np.mean(pre_nmt_accuracies['backward']):.2f} & ", end="")
    print(f"{np.mean(pre_nmt_accuracies['forward']):.2f} & ", end="")
    print(f"{np.mean(pre_nmt_accuracies['average']):.2f} & ", end="")
    print(f"{np.mean([np.mean(pre_nmt_accuracies['average']), np.mean(nmt_accuracies['average']), np.mean(ht_accuracies['average'])]):.2f} \\\\ ", end="")
    print()
    print(r"\bottomrule")
    print(r"\end{tabular}")     

elif split == 'test':

    r"""
    \begin{tabular}{ccccccc}
    \toprule
    & \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT}\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\\
    Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\
    \midrule
    en\(\leftrightarrow\)cs & tba & tba & tba & tba & tba & tba \\
    en\(\leftrightarrow\)de & tba & tba & tba & tba & tba & tba \\
    en\(\leftrightarrow\)ru & tba & tba & tba & tba & tba & tba \\
    \addlinespace
    Macro-Avg. & tba & tba & tba & tba & tba & tba & tba \\
    \bottomrule
    \end{tabular}
    """
    print(r'\begin{tabular}{ccccccc}')
    print(r'\toprule')
    print(r'& \multicolumn{3}{c}{HT} & \multicolumn{3}{c}{NMT}  \\' )
    print(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    print(r'Language Pair & \(\rightarrow\) & \(\leftarrow\) & Avg. & \(\rightarrow\) & \(\leftarrow\) & Avg. \\')
    print(r'\midrule')

    for i, lang_pair in enumerate(LANG_PAIRS):
        lp = lang_pair.split('-')
        print(lp[0]+r"\biarrow "+lp[1] + " & ", end="")
        print(f"{ht_accuracies['forward'][i]:.2f} & ", end="")
        print(f"{ht_accuracies['backward'][i]:.2f} & ", end="")
        print(f"{ht_accuracies['average'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['forward'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['backward'][i]:.2f} & ", end="")
        print(f"{nmt_accuracies['average'][i]:.2f} & ", end="")
        print(f"{np.mean([ht_accuracies['average'][i], nmt_accuracies['average'][i]]):.2f} \\\\ ", end="")
        print()

    print(r"\addlinespace")
    print(r"Macro-Avg. & ", end="")
    print(f"{np.mean(ht_accuracies['forward']):.2f} & ", end="")
    print(f"{np.mean(ht_accuracies['backward']):.2f} & ", end="")
    print(f"{np.mean(ht_accuracies['average']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['forward']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['backward']):.2f} & ", end="")
    print(f"{np.mean(nmt_accuracies['average']):.2f} & ", end="")
    print(f"{np.mean([np.mean(nmt_accuracies['average']), np.mean(ht_accuracies['average'])]):.2f} \\\\ ", end="")
    print()
    print(r"\bottomrule")
    print(r"\end{tabular}")     
