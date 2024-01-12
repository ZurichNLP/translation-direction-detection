import logging

logging.basicConfig(level=logging.INFO)

from experiments.datasets import load_all_datasets

MIN_NUM_SENTENCES_IN_DOC = 10

datasets = load_all_datasets()
datasets = [dataset for dataset in datasets if dataset.is_indirect]
lang_pairs = list(sorted(set(dataset.lang_pair for dataset in datasets)))
directions = list(sorted(set(dataset.translation_direction for dataset in datasets)))


print(r"""\begin{tabular}{lS[table-format=5.0]S[table-format=4.0]S[table-format=4.0]S[table-format=5.0]}
\toprule
Direction & \multicolumn{1}{c}{Sentence pairs} \\
\midrule""")

for lang_pair in lang_pairs:
    datasets_for_direction = [dataset for dataset in datasets if dataset.lang_pair == lang_pair]
    src_lang, tgt_lang = lang_pair.split("↔")
    src_sents = sum(dataset.num_sentences for dataset in datasets_for_direction if dataset.type == "ht")
    src_docs = sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in datasets_for_direction if dataset.type == "ht")
    ht_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "ht")
    nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "nmt")
    pre_nmt_sents = sum(dataset.num_examples for dataset in datasets_for_direction if dataset.type == "pre-nmt")
    print(f"{src_lang}\\biarrow {tgt_lang} & {src_sents} \\\\")

print(r"\addlinespace")
print(r"Total &", end=" ")
print(f"{sum(dataset.num_sentences for dataset in datasets if dataset.type == 'ht')} \\\\")
#print(f"{sum(len(dataset.get_documents(MIN_NUM_SENTENCES_IN_DOC)) for dataset in datasets if dataset.type == 'ht')} &", end=" ")
#print(f"{sum(dataset.num_examples for dataset in datasets if dataset.type == 'ht')} \\\\")
#print(f"{sum(dataset.num_examples for dataset in datasets if dataset.type == 'nmt')} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
