import logging

logging.basicConfig(level=logging.INFO)

from experiments.datasets import load_all_datasets

MIN_NUM_SENTENCES_IN_DOC = 10

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if not dataset.is_indirect]
testsets = list(sorted(set(dataset.name for dataset in datasets)))
directions = list(sorted(set(dataset.translation_direction for dataset in datasets)))

print(r"""\begin{tabular}{ccS[table-format=5.0]S[table-format=4.0]S[table-format=4.0]S[table-format=5.0]S[table-format=5.0]}
\toprule
& & \multicolumn{2}{c}{source} & \multicolumn{3}{c}{target sentences}\\
testset & direction & \multicolumn{1}{c}{sents} & \multicolumn{1}{c}{docs $\geq10$} & \multicolumn{1}{c}{HT} & \multicolumn{1}{c}{NMT} & \multicolumn{1}{c}{Pre-NMT} \\
\cmidrule(r){3-4} 	\cmidrule(l){5-7}""")

rows = []

for testset in testsets:
    for direction in directions:
        filtered_datasets = [dataset for dataset in datasets if dataset.name == testset and dataset.translation_direction == direction]
        src_lang, tgt_lang = direction.split("→")
        src_sents = sum(dataset.num_sentences for dataset in filtered_datasets if dataset.type == "ht")
        if not src_sents:
            continue
        src_docs = sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in filtered_datasets if dataset.type == "ht")
        ht_sents = sum(dataset.num_examples for dataset in filtered_datasets if dataset.type == "ht")
        nmt_sents = sum(dataset.num_examples for dataset in filtered_datasets if dataset.type == "nmt")
        pre_nmt_sents = sum(dataset.num_examples for dataset in filtered_datasets if dataset.type == "pre-nmt")
        if testset == "wmt16":
            # Italicize validation set
            rows.append(f"{testset.upper()} & {src_lang}\\textrightarrow {tgt_lang} & {src_sents} & {src_docs} & \\textit{{{ht_sents}}} & \\textit{{{nmt_sents}}} & {pre_nmt_sents if pre_nmt_sents > 0 else '-'} \\\\")
        else:
            rows.append(f"{testset.upper()} & {src_lang}\\textrightarrow {tgt_lang} & {src_sents} & {src_docs} & {ht_sents} & {nmt_sents} & {pre_nmt_sents if pre_nmt_sents > 0 else '-'} \\\\")
    if testset != testsets[-1]:
        rows.append(r"\midrule")

for row in rows:
    print(row.replace(" - ", " \multicolumn{1}{c}{-} "))

print(r"\bottomrule")
print(r"\end{tabular}")
