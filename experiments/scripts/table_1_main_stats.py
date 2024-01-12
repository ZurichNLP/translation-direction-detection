import logging

logging.basicConfig(level=logging.INFO)

from experiments.datasets import load_all_datasets

MIN_NUM_SENTENCES_IN_DOC = 10

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if not dataset.is_indirect]
validation_datasets = [dataset for dataset in datasets if (dataset.name == "wmt16" and dataset.type in ["ht", "nmt"])]
test_datasets = [dataset for dataset in datasets if (dataset.name != "wmt16" or dataset.type == "pre-nmt")]
validation_directions = list(sorted(set(dataset.translation_direction for dataset in validation_datasets)))
test_directions = list(sorted(set(dataset.translation_direction for dataset in test_datasets)))

print(r"""\begin{tabular}{lS[table-format=5.0]S[table-format=4.0]S[table-format=4.0]S[table-format=5.0]S[table-format=5.0]}
\toprule
& \multicolumn{2}{c}{source} & \multicolumn{3}{c}{target sentences}\\
direction & \multicolumn{1}{c}{sents} & \multicolumn{1}{c}{docs \footnotesize{$\geq10$}} & \multicolumn{1}{c}{HT} & \multicolumn{1}{c}{NMT} & \multicolumn{1}{c}{Pre-NMT} \\
\cmidrule(r){2-3} 	\cmidrule(l){4-6}""")

# print("Validation & & & & & \\\\")
#
# for direction in validation_directions:
#     datasets_for_direction = [dataset for dataset in validation_datasets if dataset.translation_direction == direction]
#     src_lang, tgt_lang = direction.split("→")
#     src_sents = sum(dataset.num_sentences for dataset in datasets_for_direction if dataset.type == "ht")
#     src_docs = sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in datasets_for_direction if dataset.type == "ht")
#     ht_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "ht")
#     nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "nmt")
#     pre_nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "pre-nmt")
#     print(f"{src_lang}\\textrightarrow {tgt_lang} & {src_sents} & {src_docs} & {ht_sents} & {nmt_sents} & {pre_nmt_sents if pre_nmt_sents > 0 else '-'} \\\\")
# print(r"\addlinespace")
# print(r"Total &", end=" ")
# print(f"{sum(dataset.num_sentences for dataset in validation_datasets)} &", end=" ")
# print(f"{sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in validation_datasets)} &", end=" ")
# print(f"{sum(dataset.num_examples for dataset in validation_datasets if dataset.type == 'ht')} &", end=" ")
# print(f"{sum(dataset.num_examples for dataset in validation_datasets if dataset.type == 'nmt')} &", end=" ")
# print(f"{sum(dataset.num_examples for dataset in validation_datasets if dataset.type == 'pre-nmt')} \\\\")
#
# print(r"\midrule")
# print("Test & & & & & \\\\")

for direction in test_directions:
    datasets_for_direction = [dataset for dataset in test_datasets if dataset.translation_direction == direction]
    src_lang, tgt_lang = direction.split("→")
    src_sents = sum(dataset.num_sentences for dataset in datasets_for_direction if dataset.type == "ht")
    src_docs = sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in datasets_for_direction if dataset.type == "ht")
    ht_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "ht")
    nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "nmt")
    pre_nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "pre-nmt")
    print(f"{src_lang}\\textrightarrow {tgt_lang} & {src_sents} & {src_docs} & {ht_sents} & {nmt_sents} & {pre_nmt_sents if pre_nmt_sents > 0 else '-'} \\\\".replace("-", "\multicolumn{1}{c}{-}"))
print(r"\addlinespace")
print(r"Total &", end=" ")
print(f"{sum(dataset.num_sentences for dataset in test_datasets if dataset.type == 'ht')} &", end=" ")
print(f"{sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in test_datasets if dataset.type == 'ht')} &", end=" ")
print(f"{sum(dataset.num_examples for dataset in test_datasets if dataset.type == 'ht')} &", end=" ")
print(f"{sum(dataset.num_examples for dataset in test_datasets if dataset.type == 'nmt')} &", end=" ")
print(f"{sum(dataset.num_examples for dataset in test_datasets if dataset.type == 'pre-nmt')} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
