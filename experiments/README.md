
This directory contains scripts for reproducing the experiments in the paper "Machine Translation Models are Zero-Shot Detectors of Translation Direction".

## Installation
- Follow the instructions in [README.md#Installation](../README.md) to install the package.
- `cd experiments/`
- `pip install -r requirements-exp.txt`

## Data preparation
- `cd data/`
- `wget http://data.statmt.org/wmt16/translation-task/wmt16-submitted-data-v2.tgz`
- `tar -xzf wmt16-submitted-data-v2.tgz && rm wmt16-submitted-data-v2.tgz`
- `git clone https://github.com/wmt-conference/wmt21-news-systems.git`
- `git clone https://github.com/wmt-conference/wmt22-news-systems.git`
- `git clone https://github.com/wmt-conference/wmt23-news-systems.git`
- `cd ../..`
- `python -m unittest experiments.tests.test_wmt_datasets`

## Computing the scores
We compute the translation probability scores for all datasets, languages, and models in advance and cache them in an SQLite database.

The following scripts compute the scores for the three models:
- `scripts/run_all_m2m100.py`
- `scripts/run_all_small100.py`
- `scripts/run_all_nllb.py`

By default, the computation is distributed over 12 shards so that it can be run in parallel on a multi-GPU machine. Run `python -m experiments.scripts.run_all_m2m100 0` to compute the scores in shard 0, etc.

The scores are stored in the following directory structure:
```
experiments/
└── cached_scores
    ├── nmtscore_cache0
    │   ├── alirezamsh_small100.sqlite
    │   ├── facebook_m2m100_418M.sqlite
    │   └── facebook_nllb-200-1.3B.sqlite
    ...
    └── nmtscore_cache11
        ...
```

## Reproducing the tables in the paper
The following scripts reproduce the tables in the paper:

- Table 1: `scripts/table_1_main_stats.py`
- Table 2, Tables 10–13: `scripts/full_model_table.py`
- Table 3: `scripts/table_3_accuracy_ht.py`
- Table 4: `scripts/table_4_accuracy_prenmt.py`
- Table 5: `scripts/table_5_accuracy_indirect_ht.py`
- Table 6: `scripts/table_6_examples.py`
- Table 7: `scripts/table_7_accuracy_doc_ht.py`
- Table 8: `scripts/table_8_accuracy_doc_nmt.py`
- Table 14: `scripts/full_stats_table.py`
- Table 15: `scripts/table_15_indirect_stats.py`

The following script reproduces the hypothesis test described in Section 5.6:
- `scripts/real_world_hypothesis_test.py`
