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
from experiments.supervised_baseline.scripts.model import CustomXLMRobertaForSequenceClassification, load_split

shard = sys.argv[1]
assert int(shard) in range(12)
split = sys.argv[2]
assert split in ['test', 'val']
checkpoint = sys.argv[3] 

# checkpoints
"""
checkpoints_cs-en_1e-05/checkpoint-1750
checkpoints_cs-en_1e-05/checkpoint-1400
checkpoints_cs-en_1e-05/checkpoint-700

checkpoints_ru-en_1e-05/checkpoint-1750
checkpoints_ru-en_1e-05/checkpoint-1400
checkpoints_ru-en_1e-05/checkpoint-700

checkpoints_de-en_1e-05/checkpoint-1750
checkpoints_de-en_1e-05/checkpoint-1400
checkpoints_de-en_1e-05/checkpoint-700

checkpoints_cs-en_2e-05/checkpoint-1750
checkpoints_cs-en_2e-05/checkpoint-1400
checkpoints_cs-en_2e-05/checkpoint-700

checkpoints_ru-en_2e-05/checkpoint-1750
checkpoints_ru-en_2e-05/checkpoint-1400
checkpoints_ru-en_2e-05/checkpoint-700

checkpoints_de-en_2e-05/checkpoint-1750
checkpoints_de-en_2e-05/checkpoint-1400
checkpoints_de-en_2e-05/checkpoint-700

checkpoints_cs-en_3e-05/checkpoint-1750
checkpoints_cs-en_3e-05/checkpoint-1400
checkpoints_cs-en_3e-05/checkpoint-700

checkpoints_ru-en_3e-05/checkpoint-1750
checkpoints_ru-en_3e-05/checkpoint-1400
checkpoints_ru-en_3e-05/checkpoint-700

checkpoints_de-en_3e-05/checkpoint-1750
checkpoints_de-en_3e-05/checkpoint-1400
checkpoints_de-en_3e-05/checkpoint-700

checkpoints_cs-en_dynamic/checkpoint-1750
checkpoints_cs-en_dynamic/checkpoint-1400
checkpoints_cs-en_dynamic/checkpoint-700

checkpoints_ru-en_dynamic/checkpoint-1750
checkpoints_ru-en_dynamic/checkpoint-1400
checkpoints_ru-en_dynamic/checkpoint-700

checkpoints_de-en_dynamic/checkpoint-1750
checkpoints_de-en_dynamic/checkpoint-1400
checkpoints_de-en_dynamic/checkpoint-700
"""

logging.basicConfig(level=logging.INFO)

os.environ["NMTSCORE_CACHE"] = str((Path(__file__).parent.parent / "baseline_validation_scores" / f"scores{shard}").absolute()) if split == 'val' else str((Path(__file__).parent.parent / "baseline_test_scores" / f"scores{shard}").absolute())
os.makedirs(os.environ["NMTSCORE_CACHE"], exist_ok=True)

LANG_PAIRS = ["en-cs", "de-en", "ru-en", "cs-en", "en-de", "en-ru"]

results = pd.DataFrame(columns=['source', 'target', 'type', 'predicted_label', 'gold_label'])

logging.info("Loading datasets...")
lang_pairs = {
    "cs-en": ["cs-en", "en-cs"],
    "de-en": ["de-en", "en-de"],
    "en-cs": ["cs-en", "en-cs"],
    "en-de": ["de-en", "en-de"],
    "en-ru": ["en-ru", "ru-en"],
    "ru-en": ["en-ru", "ru-en"],
}
lang_pair = checkpoint.split('_')[1]

if split == 'test':
    datasets = load_split(lang_pairs[lang_pair], split_type=split)
elif split == 'val':
    datasets = load_split(lang_pairs[lang_pair], split_type=split)

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

for lang_pair in ['cs-en', 'ru-en', 'de-en']:
    checkpoint_dir = f'experiments/supervised_baseline/{checkpoint}' # TODO: adapt to desired path to checkpoint
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
        config, tokenizer, model = checkpoint_dict[f"{dataset.src_lang}-{dataset.tgt_lang}"]
        input_src = tokenizer(source_sentence, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
        input_tgt = tokenizer(target_sentence, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
        
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

save_filename = "-".join(checkpoint_dir.split("/")[-2:])
results.to_csv(os.path.join(os.environ["NMTSCORE_CACHE"], f'{save_filename.replace("checkpoints_", "" )}.csv'))