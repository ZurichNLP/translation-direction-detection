import logging
import torch
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import sys
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from pathlib import Path

from experiments.datasets import load_wmt21_23_dataset, load_wmt16_dataset
from experiments.supervised_baseline.scripts.model import CustomXLMRobertaForSequenceClassification, load_train_val_split

shard = sys.argv[1]
assert int(shard) in range(12)
method = sys.argv[2] 
assert method in ['siamese', 'concat']
split = sys.argv[3]
assert split in ['test', 'val']
checkpoint = sys.argv[4] 

"""
checkpoint for bilingual ht and nmt training:
'checkpoint-1815'

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
if split == 'test':
    for type in ['ht', 'nmt', 'pre-nmt']:
        for wmt in ["wmt16", "wmt21", "wmt22", "wmt23"]:
            for lang_pair in LANG_PAIRS: 
                if type != 'pre-nmt':
                    if wmt in ["wmt21", "wmt22", "wmt23"] and not ((lang_pair == "de-en" and wmt == "wmt23") or (lang_pair == "cs-en" and wmt == "wmt23") or (lang_pair == "en-de" and wmt == "wmt23")):
                        datasets.append(load_wmt21_23_dataset(wmt, lang_pair, type))
                elif wmt == "wmt16" and type == 'pre-nmt':
                    ds = load_wmt16_dataset(lang_pair, type)
                    datasets.append(load_train_val_split(ds, lang_pair, split_type=split, translation_type=type))
elif split == 'val':
    for type in ['ht', 'nmt', 'pre-nmt']:
        for lang_pair in LANG_PAIRS: 
            ds = load_wmt16_dataset(lang_pair, type)
            val_set = load_train_val_split(ds, lang_pair, split_type=split, translation_type=type)
            datasets.append(val_set) if val_set else None
            print(f'{ds.translation_direction}-{type}-{split}: {val_set.num_examples}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

checkpoint_dict = {
    "cs-en": None,
    "de-en": None,
    "en-cs": None,
    "en-de": None,
    "en-ru": None,
    "ru-en": None,
}

if method == 'concat':
    checkpoint_dir = 'insert/valid/path'
    config = XLMRobertaConfig.from_pretrained(checkpoint_dir)
    tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint_dir)
    model = XLMRobertaForSequenceClassification.from_pretrained(checkpoint_dir, config=config).to(device)
    model.eval()
else:
    for lang_pair in ['cs-en', 'ru-en', 'de-en']:
        checkpoint_dir = f'experiments/supervised_baseline/models/checkpoints_{lang_pair}/siamese_ht-nmt/{checkpoint}'
        config = XLMRobertaConfig.from_pretrained(checkpoint_dir)
        tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint_dir)
        model = CustomXLMRobertaForSequenceClassification.from_pretrained(checkpoint_dir, config=config).to(device)
        model.eval()
        src, tgt = lang_pair.split('-')
        checkpoint_dict[f'{src}-{tgt}'] = config, tokenizer, model
        checkpoint_dict[f'{tgt}-{src}'] = config, tokenizer, model



for i, dataset in enumerate(datasets):
    if i % 12 != int(shard):
         continue
    print(dataset, dataset.num_examples)
    for idx, example in enumerate(tqdm(dataset.examples)):
        source_sentence = example.src
        target_sentence = example.tgt
        if method == "concat":
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
            # original direction (label: 0)
            # reverse direction (label: 1)
            if predicted_class == 0:
                predicted_label = f"{dataset.src_lang}→{dataset.tgt_lang}"
                # score for example.src->example.tgt = 1
            else:
                predicted_label = f"{dataset.tgt_lang}→{dataset.src_lang}"
                # score for example.tgt->example.src lang = 
        else:
            config, tokenizer, model = checkpoint_dict[f"{dataset.src_lang}-{dataset.tgt_lang}"]
            input_src = tokenizer(source_sentence, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
            input_tgt = tokenizer(target_sentence, return_tensors="pt", truncation=True, max_length=128, padding='max_length')

            if input_src['input_ids'].shape[1] > 512 or input_tgt['input_ids'].shape[1] > 512:
                logging.info(f"Skipping input pair with length {inputs['input_ids'].shape[1]} > 512 tokens")
                continue
            
            input_src = {key: value.to(device) for key, value in input_src.items()}
            input_tgt = {key: value.to(device) for key, value in input_tgt.items()}

            with torch.no_grad():
                outputs = model(
                    input_ids1=input_src['input_ids'],
                    attention_mask1=input_src['attention_mask'],
                    input_ids2=input_tgt['input_ids'],
                    attention_mask2=input_tgt['attention_mask']
                )
                logits = outputs.logits
            
            predicted_class1 = torch.argmax(logits, dim=-1).item()

            # reverse for assertion
            with torch.no_grad():
                outputs2 = model(
                    input_ids1=input_tgt['input_ids'],
                    attention_mask1=input_tgt['attention_mask'],
                    input_ids2=input_src['input_ids'],
                    attention_mask2=input_src['attention_mask']
                )
                logits2 = outputs2.logits
            predicted_class2 = torch.argmax(logits2, dim=-1).item()

            assert predicted_class1 == predicted_class2

            #label_mapping = {'en→cs': 0,
                            # 'cs→en': 1,
                            # 'en→de': 0,
                            # 'de→en': 1,
                            # 'en→ru': 0,
                            # 'ru→en': 1,}
            second_lang = dataset.lang_pair.replace('en↔', '').replace('↔en','')
            if predicted_class1 == 0:
                predicted_label = f"en→{second_lang}"
                # score for en->scnd lang = 1
            else:
                predicted_label = f"{second_lang}→en"
                # score for scnd->en lang = 0
            
        # TODO: cache the results instead of saving them in csv file
        assert dataset.type in ['ht', 'pre-nmt', 'nmt']
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