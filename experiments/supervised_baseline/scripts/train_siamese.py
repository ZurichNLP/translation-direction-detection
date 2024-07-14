from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from transformers import modeling_outputs
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
import torch
import random
import logging
import argparse

# Replace with actual dataset loading function
from experiments.datasets import load_wmt16_dataset

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('accelerator', type=str, help='Choice of accelerator either "cpu" for CPU or "cuda:[0-5]"')
parser.add_argument('--epochs', type=int, default=5, help='Choose number of epochs, e.g.: 3, 5 or 10')
#parser.add_argument('--lr', type=float, default=2e-5, help='Choose a learning rate, e.g.: 1e-5, 2e-5 or 3e-5')
parser.add_argument('--batch_size', type=int, default=16, help='Choose a batch size, e.g.: 4, 8 or 16')
parser.add_argument('--train_type', nargs='+', default=['ht', 'nmt'], help='Choose translation type of the training data, e.g.: ht nmt or ht')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set device based on user input
if torch.cuda.is_available() and args.accelerator.startswith('cuda'):
    print(f'GPU available and set to {args.accelerator}')
    device = torch.device(args.accelerator)
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#print(f'Learning rate is {args.lr}.')
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

class CustomXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    def forward(
        self,
        input_ids1: Optional[torch.LongTensor] = None,
        attention_mask1: Optional[torch.FloatTensor] = None,
        input_ids2: Optional[torch.LongTensor] = None,
        attention_mask2: Optional[torch.FloatTensor] = None,
        token_type_ids1: Optional[torch.LongTensor] = None,
        token_type_ids2: Optional[torch.LongTensor] = None,
        position_ids1: Optional[torch.LongTensor] = None,
        position_ids2: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.roberta(
            input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            position_ids=position_ids1,
        )
        sequence_output1 = outputs1.last_hidden_state  # or maybe pooled output?

        outputs2 = self.roberta(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids2,
        )
        sequence_output2 = outputs2.last_hidden_state  # or maybe pooled output?

        assert sequence_output1.shape == sequence_output2.shape, \
            f"Shape mismatch: {sequence_output1.shape} vs {sequence_output2.shape}"

        combined_representation = sequence_output1 + sequence_output2

        logits = self.classifier(combined_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs1[2:] + outputs2[2:]
            return ((loss,) + output) if loss is not None else output

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(sequence_output1, sequence_output2),
            attentions=None,
        )

# load data
logging.info("Loading datasets...")
datasets = []
sum_examples = 0
for type in args.train_type:
    for lang_pair in ["cs-en", "de-en", "en-cs", "en-de", "en-ru", "ru-en"]:
        ds = load_wmt16_dataset(lang_pair, type)
        datasets.append(ds)
        if ds.num_examples > 1500:
            random.shuffle(ds.examples)
            ds.examples = ds.examples[:1500]
        sum_examples += ds.num_examples
        print(f'{ds.translation_direction}-{type}: {ds.num_examples}')
print(f'sum of examples: {sum_examples}')


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
    f'experiments/supervised_baseline/models/checkpoints/siamese_{"-".join(args.train_type)}', 
    evaluation_strategy='no',
    save_strategy='epoch',       
    #learning_rate=args.lr, 
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
