
from transformers import XLMRobertaForSequenceClassification
from transformers import modeling_outputs
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
import torch
import random

from experiments.datasets import load_wmt16_dataset, TranslationDataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_train_val_split(dataset, lang_pair, split_type, translation_type):
    set_seed(42)

    random.shuffle(dataset.examples)

    dataset.examples = dataset.examples[:1500] if dataset.num_examples > 1500 else dataset.examples

    # validation
    val_set_examples = dataset.examples[:50]
    val_set = TranslationDataset(
        name='wmt16',
        type=translation_type,
        src_lang=dataset.src_lang,
        tgt_lang=dataset.tgt_lang,
        examples=val_set_examples
    )

    if translation_type == 'pre-nmt':
        test_set_examples = dataset.examples[50:]
        test_set = TranslationDataset(
            name='wmt16',
            type=translation_type,
            src_lang=dataset.src_lang,
            tgt_lang=dataset.tgt_lang,
            examples=test_set_examples
        )
    else:
        # training
        train_set_examples = dataset.examples[50:]
        train_set = TranslationDataset(
            name='wmt16',
            type=translation_type,
            src_lang=dataset.src_lang,
            tgt_lang=dataset.tgt_lang,
            examples=train_set_examples
        )

    if split_type == 'train':
        return train_set
    elif split_type == 'val':
        return val_set
    elif split_type == 'test' and translation_type == 'pre-nmt':
        return test_set
    
class CustomXLMRobertaClassificationHead(RobertaClassificationHead):
    def __init__(self, config):
        super().__init__(config)
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)  # Input size: 3072
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features, **kwargs):
        x = features # since the features are already features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class CustomXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomXLMRobertaClassificationHead(config)
    
    
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
        # hidden_states1 = outputs1.hidden_states  # last hidden state
        # hidden_states_tensor1 = torch.stack(hidden_states1, dim=0)
        # summed_hidden_states1 = hidden_states_tensor1.sum(dim=0)
        # sequence_output1 = summed_hidden_states1.mean(dim=1) # avg of the hidden states
        sequence_output1 = outputs1.last_hidden_state[:, 0, :] # pooler output


        outputs2 = self.roberta(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids2,
        )
        # hidden_states2 = outputs2.hidden_states  # last hidden state
        # hidden_states_tensor2 = torch.stack(hidden_states2, dim=0)
        # summed_hidden_states2 = hidden_states_tensor2.sum(dim=0)
        # sequence_output2 = summed_hidden_states2.mean(dim=1) # avg of the hidden states
        sequence_output2 = outputs2.last_hidden_state[:, 0, :]  # pooler output
        
        #print(sequence_output1.shape, sequence_output2.shape)
        assert sequence_output1.shape == sequence_output2.shape, \
            f"Shape mismatch: {sequence_output1.shape} vs {sequence_output2.shape}"

        diff_embs = torch.abs(sequence_output1 - sequence_output2)
        prod_embs = sequence_output1 * sequence_output2
        
        #combined_representation = torch.cat((sequence_output1, sequence_output2, diff_embs, prod_embs), dim=1) # introduces dependence on order?
        combined_representation = sequence_output1 + sequence_output2 + diff_embs + prod_embs

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