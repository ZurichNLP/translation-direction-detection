
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

## Reproducing the baseline
To train the supervised systems on WMT data to reproduce our baseline run the following:
- `python -m experiments.supervised_baseline.scripts.train cuda:0 cs-en --lr 1e-05 --dataset wmt`
- `python -m experiments.supervised_baseline.scripts.train cuda:0 ru-en --lr 1e-05 --dataset wmt`
- `python -m experiments.supervised_baseline.scripts.train cuda:0 de-en --lr 1e-05 --dataset wmt`

Afterwards, use the resulting checkpoints to predict the labels for the test and/or validation set, where the computation is distributed over shards as above:

- `python -m experiments.scripts.run_baseline checkpoints_cs-en_1e-05/checkpoint-700 [0-11] [val/test] wmt wmt`
- `python -m experiments.scripts.run_baseline checkpoints_ru-en_1e-05/checkpoint-700 [0-11] [val/test] wmt wmt`
- `python -m experiments.scripts.run_baseline checkpoints_de-en_1e-05/checkpoint-1400 [0-11] [val/test] wmt wmt`

The labeled segment pairs are stored in the following directory structure:
```
experiments/
└── supervised_baseline/wmt/baseline_[validation/test]_scores/
    ├── scores0
    │   ├── cs-en_1e-05-checkpoint-700.csv
    │   ├── ru-en_1e-05-checkpoint-700.csv
    │   └── de-en_1e-05-checkpoint-1400.csv
    ...
    └── scores11
        ...
```

To reproduce the experiments with the supervised systems that were trained on Europarl data, first, download the Europarl corpus and use the EuroparlExtract package to sort it by translation direction as described in the [README]([url](https://github.com/mustaszewski/europarl-extract?tab=readme-ov-file#europarlextract)) for the language pairs en-de, en-cs and de-fr. Then, move the resulting datasets into the ```experiments/supervised_baseline/data/parallel/``` folder. The directory structure should look as follows:
```
experiments/
└── supervised_baseline/europarl/data/parallel/
    ├── CS-EN/tab
    │   ├── 07-09-03-016_082_cs-en.tab
    │   ├── 07-09-03-017_105_cs-en.tab
    │   ...
    ├── DE-EN/tab
        ...
    ├── DE-FR/tab
        ...
    ├── EN-CS/tab
        ...
    ├── EN-DE/tab
        ...
    ├── FR-DE/tab
        ...
```

Then train the systems as follows and make sure to comment out line 135 in `experiments.supervised_baseline.scripts.train` for cs-en and de-fr:
- `python -m experiments.supervised_baseline.scripts.train cuda:0 cs-en --dataset europarl`
- `python -m experiments.supervised_baseline.scripts.train cuda:0 de-fr --dataset europarl`
- `python -m experiments.supervised_baseline.scripts.train cuda:0 de-en --lr 1e-05 --dataset europarl`

Afterwards, use the resulting checkpoints to predict the labels for the test and/or validation set, where the computation is distributed over shards as above. When running this on the test set, you can choose between the WMT and the Europarl based test sets:
- `python -m experiments.scripts.run_baseline checkpoints_cs-en_dynamic_20498/checkpoint-6410 [0-11] [val/test] europarl [wmt/europarl]`
- `python -m experiments.scripts.run_baseline checkpoints_ru-en_dynamic_20498/checkpoint-6410 [0-11] [val/test] europarl [wmt/europarl]`
- `python -m experiments.scripts.run_baseline checkpoints_de-en_1e-05_20498/checkpoint-6410 [0-11] [val/test] europarl [wmt/europarl]`

The labeled segment pairs are stored in the following directory structure:
```
experiments/
└── supervised_baseline/europarl/europarl_[wmt_]baseline_[validation/test]_scores/
    ├── scores0
    │   ├── cs-en_dynamic-checkpoint-6410.csv
    │   ├── ru-en_dynamic-checkpoint-6410.csv
    │   └── de-en_1e-05-checkpoint-6410.csv
    ...
    └── scores11
        ...
```

## Reproducing the tables in the paper
The following scripts reproduce the tables in the paper:

- Table 1: `scripts/table_1_main_stats.py`
- Table 2: `scripts/table_2_baseline_comparison.py`
- Table 3, Tables 10–13: `scripts/full_model_table.py`
- Table 4: `scripts/table_4_accuracy_ht.py`
- Table 5: `scripts/table_5_accuracy_prenmt.py`
- Table 6: `scripts/table_6_accuracy_indirect_ht.py`
- Table 7: `scripts/table_7_examples.py`
- Table 8: `scripts/table_8_accuracy_doc_ht.py`
- Table 9: `scripts/table_9_accuracy_doc_nmt.py`
- Table 14-16: `scripts/accuracy_baseline.py`
- Table 17: `scripts/accuracy_wmt.py`
- Table 18: `scripts/full_stats_table.py`
- Table 19: `scripts/indirect_stats.py`

The following script reproduces the hypothesis test described in Section 5.6:
- `scripts/real_world_hypothesis_test.py`
