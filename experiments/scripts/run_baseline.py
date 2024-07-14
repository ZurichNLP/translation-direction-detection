import logging
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import sys
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from pathlib import Path

from experiments.datasets import load_wmt21_23_dataset, load_wmt16_dataset

shard = sys.argv[1]
assert int(shard) in range(12)
method = sys.argv[2] 
checkpoint_dir = sys.argv[3]
"""
paths to checkpoints:
'experiments/supervised_baseline/models/checkpoints/concat/checkpoint-5625' for concat and HT training data 
'experiments/supervised_baseline/models/checkpoints/siamese/checkpoint-2815' for siamese and HT training data 
'experiments/supervised_baseline/models/checkpoints/siamese_ht-nmt/checkpoint-5625' for siamese and HT and NMT training data 
"""

logging.basicConfig(level=logging.INFO)

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "cached_scores" / f"nmtscore_cache{shard}").absolute())
os.makedirs(os.environ["NMTSCORE_CACHE"], exist_ok=True)

LANG_PAIRS = ["en-cs", "de-en", "ru-en", "cs-en", "en-de", "en-ru"]

results = pd.DataFrame(columns=['source', 'target', 'type', 'predicted_label', 'gold_label'])

datasets = []
for type in ['ht', 'nmt', 'pre-nmt']:
    for wmt in ["wmt16", "wmt21", "wmt22", "wmt23"]:
        for lang_pair in LANG_PAIRS: 
            if type != 'pre-nmt':
                if wmt in ["wmt21", "wmt22", "wmt23"] and not ((lang_pair == "de-en" and wmt == "wmt23") or (lang_pair == "cs-en" and wmt == "wmt23")):
                    datasets.append(load_wmt21_23_dataset(wmt, lang_pair, type))
            elif wmt == "wmt16" and type == 'pre-nmt':
                datasets.append(load_wmt16_dataset(lang_pair, type))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

config = AutoConfig.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, config=config).to(device)
model.eval()

for i, dataset in enumerate(datasets):
    if i % 12 != int(shard):
         continue
    print(dataset, dataset.num_examples)
    for idx, example in enumerate(tqdm(dataset.examples)):
        source_sentence = example.src
        target_sentence = example.tgt
        inputs = tokenizer(source_sentence, target_sentence, return_tensors="pt")

        if inputs['input_ids'].shape[1] > 512:
            logging.info(f"Skipping input pair with length {inputs['input_ids'].shape[1]} > 512 tokens")
            continue

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class = torch.argmax(logits, dim=-1).item()

        # map class number to class label
        if method == 'concat':
            # original direction (label: 0)
            # reverse direction (label: 1)
            if predicted_class == 0:
                predicted_label = f"{dataset.src_lang}→{dataset.tgt_lang}"
                # score for example.src->example.tgt = 1
            else:
                predicted_label = f"{dataset.tgt_lang}→{dataset.src_lang}"
                # score for example.tgt->example.src lang = 
        else:
            #label_mapping = {'en→cs': 0,
                            # 'cs→en': 1,
                            # 'en→de': 0,
                            # 'de→en': 1,
                            # 'en→ru': 0,
                            # 'ru→en': 1,}
            second_lang = dataset.lang_pair.replace('en↔', '').replace('↔en','')
            if predicted_class == 0:
                predicted_label = f"en→{second_lang}"
                # score for en->scnd lang = 1
            else:
                predicted_label = f"{second_lang}→en"
                # score for scnd->en lang = 0

        # TODO: cache the results instead of saving them in csv file
        result = {
            'source': example.src,
            'target': example.tgt,
            'type': dataset.type,
            'predicted_label': predicted_label,
            'gold_label': dataset.translation_direction,
        }

        results.loc[len(results)] = result
    
print(len(results.index))

results.to_csv(os.path.join(os.environ["NMTSCORE_CACHE"], f'baseline_{"_".join(checkpoint_dir.split("/")[-2:])}.csv'))