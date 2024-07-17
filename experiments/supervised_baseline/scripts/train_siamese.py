from transformers import XLMRobertaTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
import torch
import random
import logging
import argparse

from experiments.supervised_baseline.scripts.model import CustomXLMRobertaForSequenceClassification, load_split, set_seed

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('accelerator', type=str, help='Choice of accelerator either "cpu" for CPU or "cuda:[0-5]"')
parser.add_argument('lang_pair', type=str, help='Choose language pair to train on from "cs-en", "de-en" and "ru-en".')
parser.add_argument('--epochs', type=int, default=5, help='Choose number of epochs, e.g.: 3, 5 or 10')
parser.add_argument('--lr', type=float, default=2e-5, help='Choose a learning rate, e.g.: 1e-5, 2e-5 or 3e-5')
parser.add_argument('--batch_size', type=int, default=16, help='Choose a batch size, e.g.: 4, 8 or 16')
parser.add_argument('--train_type', nargs='+', default=['ht', 'nmt'], help='Choose translation type of the training data, e.g.: ht nmt or ht')
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
print(f'Training data type: {args.train_type}')
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
datasets = []
sum_examples_train = 0
sum_examples_val = 0
lang_pairs = {
    "cs-en": ["cs-en", "en-cs"],
    "de-en": ["de-en", "en-de"],
    "en-cs": ["cs-en", "en-cs"],
    "en-de": ["de-en", "en-de"],
    "en-ru": ["en-ru", "ru-en"],
    "ru-en": ["en-ru", "ru-en"],
}
for type in args.train_type + ['pre-nmt']:
    for lp in lang_pairs[args.lang_pair]:
        train_set = load_split(lp, split_type='train', translation_type=type, wmt_type='wmt16') if type != 'pre-nmt' else None
        sum_examples_train += train_set.num_examples if type != 'pre-nmt' else 0
        datasets.append(train_set) if train_set else None
print(f'sum of training examples: {sum_examples_train}')


# preprocess
label_mapping = {'en→cs': 0,
                 'cs→en': 1,
                 'en→de': 0,
                 'de→en': 1,
                 'en→ru': 0,
                 'ru→en': 1,}

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
model = CustomXLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model.to(device)

tokenizer = XLMRobertaTokenizer.from_pretrained(model_checkpoint)

tokenized_data_src = tokenizer(source_side, truncation=True, padding='max_length', max_length=128, return_tensors='pt') #input_ids, token_type_ids, attention_mask
tokenized_data_ref = tokenizer(ref_side, truncation=True, padding='max_length', max_length=128, return_tensors='pt') #input_ids, token_type_ids, attention_mask

training_data = CustomDataset(tokenized_data_src, tokenized_data_ref, labels)

training_args = TrainingArguments(
    f'experiments/supervised_baseline/models/checkpoints_{args.lang_pair}_pooler/siamese_{"-".join(args.train_type)}', 
    evaluation_strategy='no',
    save_strategy='epoch',       
    learning_rate=args.lr, 
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    weight_decay=0.01,       
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

trainer.train()