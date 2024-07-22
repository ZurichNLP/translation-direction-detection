
from transformers import XLMRobertaForSequenceClassification
from transformers import modeling_outputs
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
import torch
import random


from experiments.datasets import load_wmt16_dataset, load_wmt21_23_dataset, TranslationDataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(lang_pairs, split_type):
    set_seed(42)
    val_datasets = []
    train_datasets = []
    test_datasets = []

    for lang_pair in lang_pairs:
        ht = load_wmt16_dataset(lang_pair, 'ht')
        nmt = load_wmt16_dataset(lang_pair, 'nmt')
        prenmt = load_wmt16_dataset(lang_pair, 'pre-nmt')

        val_exmpls_ht = []
        val_exmpls_nmt = []
        val_exmpls_prenmt = []

        train_exmpls_ht = []
        train_exmpls_nmt = []
        train_exmpls_prenmt = []

        test_exmpls_prenmt = []

        for ex_ht in ht.examples:
            for ex_nmt in nmt.examples:
                if len(val_exmpls_ht) < 100:
                    if ex_nmt.src == ex_ht.src:
                        val_exmpls_ht.append(ex_ht)
                        val_exmpls_nmt.append(ex_nmt)
                        break
                elif len(train_exmpls_ht) < 1500:
                    if ex_nmt.src == ex_ht.src:
                        train_exmpls_ht.append(ex_ht)
                        train_exmpls_nmt.append(ex_nmt)
                        break
                else:
                    break

        for ex_ht in val_exmpls_ht:
            for ex_prenmt in prenmt.examples:
                if ex_ht.src == ex_prenmt.src:
                    val_exmpls_prenmt.append(ex_prenmt)
                    break
        
        for ex_ht in train_exmpls_ht:
            for ex_prenmt in prenmt.examples:
                if ex_ht.src == ex_prenmt.src:
                    train_exmpls_prenmt.append(ex_prenmt)
                    break
        
        for ex_prenmt in prenmt.examples[:-1]:
            if ex_prenmt in val_exmpls_prenmt:
                continue
            if ex_prenmt in train_exmpls_prenmt:
                continue
            else:
                test_exmpls_prenmt.append(ex_prenmt)

        print(len(val_exmpls_ht), len(val_exmpls_nmt), len(val_exmpls_prenmt))
        assert len(val_exmpls_ht) == len(val_exmpls_nmt) == len(val_exmpls_prenmt) == 100
        print(len(train_exmpls_ht), len(train_exmpls_nmt), len(train_exmpls_prenmt))
        assert len(train_exmpls_ht) == len(train_exmpls_nmt) == len(train_exmpls_prenmt) <= 1500
        for i in range(0, len(val_exmpls_ht)):
            assert val_exmpls_ht[i].src == val_exmpls_nmt[i].src == val_exmpls_prenmt[i].src
        for i in range(0, len(train_exmpls_ht)):
            assert train_exmpls_ht[i].src == train_exmpls_nmt[i].src == train_exmpls_prenmt[i].src
       
        for _ in [ht, nmt, prenmt]:

            if _.type == 'ht':
                val_exmpls = val_exmpls_ht
                train_exmpls = train_exmpls_ht
            elif _.type == 'nmt':
                val_exmpls = val_exmpls_nmt
                train_exmpls = train_exmpls_nmt    
            elif _.type == 'pre-nmt':
                val_exmpls = val_exmpls_prenmt
                train_exmpls = train_exmpls_prenmt
                test_exmpls = test_exmpls_prenmt        

            val_set = TranslationDataset(
            name='wmt16',
            type=_.type,
            src_lang=_.src_lang,
            tgt_lang=_.tgt_lang,
            examples=val_exmpls
                )

            val_datasets.append(val_set)

            if _.type != 'pre-nmt':

                random.shuffle(train_exmpls)

                train_set = TranslationDataset(
                name='wmt16',
                type=_.type,
                src_lang=_.src_lang,
                tgt_lang=_.tgt_lang,
                examples=train_exmpls
                    )

                train_datasets.append(train_set)

                test_set = TranslationDataset(
                name='wmt16',
                type=_.type,
                src_lang=_.src_lang,
                tgt_lang=_.tgt_lang,
                examples=test_exmpls
                    )

                test_datasets.append(test_set)
        
        for wmt_year in ['wmt21', 'wmt22', 'wmt23']:
            if wmt_year == 'wmt23' and lang_pair in ['de-en', 'en-de', 'cs-en']:
                continue
            else:
                ht = load_wmt21_23_dataset(wmt_year, lang_pair, 'ht')
                nmt = load_wmt21_23_dataset(wmt_year, lang_pair, 'nmt')
                test_datasets.append(ht)
                test_datasets.append(nmt)
        
    if split_type == 'val':
        print(f'number of validation sets: {len(val_datasets)}')
        for ds in val_datasets:
            print(f'{ds.translation_direction}: {ds.num_examples}')
        return val_datasets 
    
    elif split_type == 'train':
        print(f'number of training sets: {len(train_datasets)}')
        for ds in train_datasets:
            print(f'{ds.translation_direction}: {ds.num_examples}')
        return train_datasets
        
    elif split_type == 'test':
        print(f'number of test sets: {len(test_datasets)}')
        for ds in test_datasets:
            print(f'{ds.translation_direction}: {ds.num_examples}')
        return test_datasets
    

class CustomXLMRobertaClassificationHead(RobertaClassificationHead):
    def __init__(self, config):
        super().__init__(config)
        self.dense = torch.nn.Linear(3 * config.hidden_size, config.hidden_size) 
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

        outputs2 = self.roberta(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids2,
        )

        """
        # avg of hidden states
        hidden_states1 = outputs1.hidden_states
        masked_hidden_states1 = hidden_states1 * attention_mask1.unsqueeze(-1)
        sum_hidden_states1 = masked_hidden_states1.sum(dim=1)
        count_valid_positions1 = attention_mask1.sum(dim=1, keepdim=True)
        sequence_output1 = sum_hidden_states1 / count_valid_positions1 # avg

        hidden_states2 = outputs2.hidden_states
        masked_hidden_states2 = hidden_states2 * attention_mask2.unsqueeze(-1)
        sum_hidden_states2 = masked_hidden_states2.sum(dim=1)
        count_valid_positions2 = attention_mask2.sum(dim=1, keepdim=True)
        count_valid_positions2 = torch.max(count_valid_positions2, torch.ones_like(count_valid_positions2))
        sequence_output2 = sum_hidden_states2 / count_valid_positions2 # avg
        """
        
        # pooler output
        sequence_output1 = outputs1.last_hidden_state[:, 0, :] 
        sequence_output2 = outputs2.last_hidden_state[:, 0, :]
        
        
        #print(sequence_output1.shape, sequence_output2.shape)
        assert sequence_output1.shape == sequence_output2.shape, \
            f"Shape mismatch: {sequence_output1.shape} vs {sequence_output2.shape}"

        # comet sentence representation calculations
        diff_embs = torch.abs(sequence_output1 - sequence_output2)
        prod_embs = sequence_output1 * sequence_output2
        
        # concatenation of representations
        combined_representation = torch.cat((sequence_output1 + sequence_output2, diff_embs, prod_embs), dim=1) # addition to not introduce order
        # combined_representation = sequence_output1 + sequence_output2 + diff_embs + prod_embs

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