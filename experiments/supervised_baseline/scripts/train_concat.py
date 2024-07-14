from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
import random
import logging
import argparse

from experiments.datasets import load_wmt16_dataset 

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('accelerator', type=str,
                    help='Choose accelerator either "cpu" for CPU or "cuda:[0-5]') # choice gpu or cpu
parser.add_argument('--epochs', type=int, default=5,
                    help='Choose number of epochs, e.g.: 3, 5 or 10')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='Choose a learning rate, e.g.: 1e-5, 2e-5 or 3e-5')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Choose a batch size, e.g.: 4, 8 or 16')
parser.add_argument('--lang_pairs', nargs='+', default=["cs-en", "de-en", "en-cs", "en-de", "en-ru", "ru-en"],
                    help='Specify language pairs, e.g., cs-en de-en')
parser.add_argument('--data_types', nargs='+', default=["ht"],
                    help='Specify data types, e.g., ht, nmt')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# check for cuda
print('\n')
if torch.cuda.is_available() and args.accelerator.startswith('cuda'): 
    print(f'GPU available and set to {args.accelerator}')
    device = torch.device(args.accelerator)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print(f'Chosen learning rate is {args.lr}.')
print(f'Chosen number of epochs is {args.epochs}.')
print(f'Batch size is {args.batch_size}')
print('\n')

model_checkpoint = 'xlm-roberta-base'
tokenizer = XLMRobertaTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# concatenated version
# preprocess
# all data that we have is in original direction only (label: 0)
# to have reverse samples as well, all sentence pairs are concatenated in reverse order as well (label: 1)

# load data
logging.info("Loading datasets...")
datasets = []
for type in args.data_types:
        for lang_pair in args.lang_pairs:
            datasets.append(load_wmt16_dataset(lang_pair, type))

# preprocess
source_side = []
ref_side = []
labels = []

for d in datasets:
    print(d.name, d.translation_direction, d.lang_pair, d.type, d.num_examples, d.num_sentences)
    print(set([e.sysid for e in d.examples]))
    for e in d.examples:
         source_side.append(e.src) # original direction (label: 0)
         source_side.append(e.tgt) # reverse direction (label: 1)
         ref_side.append(e.tgt) # original direction (label: 0)
         ref_side.append(e.src) # reverse direction (label: 1)
         labels.append(0)
         labels.append(1)


assert len(source_side) == len(ref_side) == len(labels)

labels = torch.tensor(labels).to(device) 
num_labels = len(torch.unique(labels))

assert num_labels == 2

tokenized_data_pair = tokenizer(source_side, ref_side, truncation=True, padding='max_length', max_length=128, return_tensors='pt') #input_ids, token_type_ids, attention_mask
tokenized_data_pair = {key: val.to(device) for key, val in tokenized_data_pair.items()}  # Move tokenized data to GPU if available

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx].to(device)) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

training_data = CustomDataset(tokenized_data_pair, labels)

# load model
model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model.to(device)

training_args = TrainingArguments(
    "experiments/supervised_baseline/models/checkpoints/concat",
    evaluation_strategy = 'no',
    save_strategy = 'epoch',
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    per_device_train_batch_size = args.batch_size,
    weight_decay=0.01,
)

trainer = Trainer(
                  model=model, 
                  args=training_args, 
                  train_dataset=training_data,
                  tokenizer=tokenizer 

                  )

trainer.train()
