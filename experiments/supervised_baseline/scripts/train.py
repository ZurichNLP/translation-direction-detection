from transformers import XLMRobertaTokenizer, TrainingArguments, Trainer, XLMRobertaForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from typing import List, Optional, Tuple, Union
import torch
import logging
import argparse

from experiments.supervised_baseline.scripts.model import CustomXLMRobertaForSequenceClassification, load_split, set_seed, load_split_europarl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('accelerator', type=str, help='Choice of accelerator either "cpu" for CPU or "cuda:[0-5]"')
parser.add_argument('lang_pair', type=str, help='Choose language pair to train on from "cs-en", "de-en" and "ru-en"/"fr-de" (depending on whether you train on WMT or Europarl data).')
parser.add_argument('--epochs', type=int, default=5, help='Choose number of epochs, e.g.: 3, 5 or 10')
parser.add_argument('--lr', type=float, default='dynamic', help='Choose a learning rate, e.g.: 1e-5, 2e-5 or 3e-5')
parser.add_argument('--batch_size', type=int, default=16, help='Choose a batch size, e.g.: 4, 8 or 16')
parser.add_argument('--dataset', type=str, default='wmt', help='Choose a dataset to train on, e.g.: wmt or europarl')
parser.add_argument('--architecture', type=str, default='siamese', help='Choose a architecture to train on, e.g.: concat or siamese')
args = parser.parse_args()

set_seed(42)

# Set device based on user input
if torch.cuda.is_available() and args.accelerator.startswith('cuda'):
    print(f'GPU available and set to {args.accelerator}')
    device = torch.device(args.accelerator)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print(f'Learning rate is {args.lr}.')
print(f'Number of epochs are {args.epochs}.')
print(f'Batch size is {args.batch_size}')
print(f'Language pair is {args.lang_pair}')
print('\n')

class CustomDataset(Dataset):
    def __init__(self, tokenized_data_src, tokenized_data_ref, labels):
        self.tokenized_data_src = tokenized_data_src
        self.tokenized_data_ref = tokenized_data_ref
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs1 = {key: val[idx] for key, val in self.tokenized_data_src.items()}
        inputs2 = {key: val[idx] for key, val in self.tokenized_data_ref.items()}
        label = self.labels[idx]
        
        return {
            'input_ids1': inputs1['input_ids'],
            'attention_mask1': inputs1['attention_mask'],
            'input_ids2': inputs2['input_ids'],
            'attention_mask2': inputs2['attention_mask'],
            'labels': label
        }

def collate_fn(batch):
    input_ids1 = torch.stack([item['input_ids1'] for item in batch])
    attention_mask1 = torch.stack([item['attention_mask1'] for item in batch])
    input_ids2 = torch.stack([item['input_ids2'] for item in batch])
    attention_mask2 = torch.stack([item['attention_mask2'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids1': input_ids1,
        'attention_mask1': attention_mask1,
        'input_ids2': input_ids2,
        'attention_mask2': attention_mask2,
        'labels': labels
    }


# load data
logging.info("Loading datasets...")
lang_pairs = {
    "cs-en": ["cs-en", "en-cs"],
    "de-en": ["de-en", "en-de"],
    "en-cs": ["cs-en", "en-cs"],
    "en-de": ["de-en", "en-de"],
    "en-ru": ["en-ru", "ru-en"],
    "ru-en": ["en-ru", "ru-en"],
    "de-fr": ["de-fr", "fr-de"],
}

datasets = load_split(lang_pairs[args.lang_pair], split_type='train') if args.dataset == 'wmt' else load_split_europarl(lang_pairs[args.lang_pair], split_type='train')

# preprocess
label_mapping = {'en→cs': 0,
                 'cs→en': 1,
                 'en→de': 0,
                 'de→en': 1,
                 'en→ru': 0,
                 'ru→en': 1,
                 'fr→de': 0, 
                 'de→fr': 1}

source_side = []
ref_side = []
labels = []

for d in datasets:
    for e in d.examples:
        source_side.append(e.src)
        ref_side.append(e.tgt)
        labels.append(label_mapping[d.translation_direction])

assert len(source_side) == len(ref_side) == len(labels)

labels = torch.tensor(labels)
num_labels = len(torch.unique(labels))

assert num_labels == 2

model_checkpoint = 'xlm-roberta-base'
if args.architecture == 'concat':
    model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
else:
    model = CustomXLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

model.to(device)

tokenizer = XLMRobertaTokenizer.from_pretrained(model_checkpoint)

if args.architecture == 'concat':
    """
    # this approach results in models that rely heavily on the input order of the sentences
    tokenized_data = tokenizer(source_side, ref_side, truncation=True, padding='max_length', max_length=256, return_tensors='pt') #input_ids, token_type_ids, attention_mask
    tokenized_data["labels"] = torch.tensor(labels)
    data_dict = {key: tokenized_data[key].tolist() for key in tokenized_data}
    training_data = HFDataset.from_dict(data_dict)
    """

    """
    # this approach flips the input order and flips the labels for the second half of the dataset in order to avoit the model to rely on the input order
    # this approach results in a model that is biased towards one translation direction
    half = len(source_side) // 2
    first_half = source_side[:half] + ref_side[half:]
    second_half = ref_side[:half] + source_side[half:]
    tokenized_data = tokenizer(first_half, second_half, truncation=True, padding='max_length', max_length=256, return_tensors='pt') #input_ids, token_type_ids, attention_mask
    tokenized_data["labels"] = torch.cat([
        labels[:half],
        1 - labels[half:]  # Flip labels for reversed order
    ])
    data_dict = {key: tokenized_data[key].tolist() for key in tokenized_data}
    training_data = HFDataset.from_dict(tokenized_data)
    """

    # this approach concats the src and ref once in one dir and once in the other dir and both is used for training
    # this model is then trained on double the data than the other approaches
    # this approach results in models that rely on the input order of the sentences
    src_side_both_dirs = []
    ref_side_both_dirs = []
    labels_both_dirs = []

    for src, ref in zip(source_side, ref_side):
        src_side_both_dirs.append(src)
        ref_side_both_dirs.append(ref)
        src_side_both_dirs.append(ref)
        ref_side_both_dirs.append(src)
        labels_both_dirs.append(label_mapping[d.translation_direction])
        labels_both_dirs.append(1 - label_mapping[d.translation_direction])
        
    tokenized_data = tokenizer(src_side_both_dirs, ref_side_both_dirs, truncation=True, padding='max_length', max_length=256, return_tensors='pt') #input_ids, token_type_ids, attention_mask
    tokenized_data["labels"] = torch.tensor(labels_both_dirs)
    data_dict = {key: tokenized_data[key].tolist() for key in tokenized_data}
    training_data = HFDataset.from_dict(data_dict)


else:
    tokenized_data_src = tokenizer(source_side, truncation=True, padding='max_length', max_length=128, return_tensors='pt') #input_ids, token_type_ids, attention_mask
    tokenized_data_ref = tokenizer(ref_side, truncation=True, padding='max_length', max_length=128, return_tensors='pt') #input_ids, token_type_ids, attention_mask
    training_data = CustomDataset(tokenized_data_src, tokenized_data_ref, labels)

if args.lr != 'dynamic':
    training_args = TrainingArguments(
        f'experiments/supervised_baseline/europarl/checkpoints_{args.lang_pair}_dynamic_{len(source_side)}' if args.dataset == "europarl" else f'experiments/supervised_baseline/wmt/checkpoints_{args.lang_pair}_{args.lr}', 
        evaluation_strategy='no',
        save_strategy='epoch',       
        learning_rate=args.lr, 
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
        lr_scheduler_type='constant',
    )
else:
    training_args = TrainingArguments(
        f'experiments/supervised_baseline/europarl/checkpoints_{args.lang_pair}_dynamic_{len(source_side)}' if args.dataset == "europarl" else f'experiments/supervised_baseline/wmt/checkpoints_{args.lang_pair}_dynamic', 
        evaluation_strategy='no',
        save_strategy='epoch',       
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    tokenizer=tokenizer,
    data_collator=collate_fn if not args.architecture == 'concat' else None,
)

trainer.train()
