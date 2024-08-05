
from transformers import XLMRobertaForSequenceClassification
from transformers import modeling_outputs
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
import torch
import random
import os


from experiments.datasets import load_wmt16_dataset, load_wmt21_23_dataset, TranslationDataset, TranslationExample

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
        if lang_pair not in ['de-fr', 'fr-de']:
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
                
                else:

                    test_set = TranslationDataset(
                    name='wmt16',
                    type=_.type,
                    src_lang=_.src_lang,
                    tgt_lang=_.tgt_lang,
                    examples=test_exmpls
                        )

                    test_datasets.append(test_set)
            
        for wmt_year in ['wmt21', 'wmt22', 'wmt23']:
            if wmt_year == 'wmt23' and lang_pair in ['de-en', 'en-de', 'cs-en', 'fr-de', 'de-fr']:
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


def load_split_europarl(lang_pairs, split_type):
    set_seed(42)

    val_datasets = []
    train_datasets = []
    test_datasets = []

    path_to_data = "experiments/supervised_baseline/data/parallel/"
    """  
    num_examples_per_dir = {
            'cs-en' : 12812,
            'de-en' : 267049,
            'fr-de' : 222888,            
            'en-cs' : 12812,
            'en-de' : 267049,
            'de-fr' : 222888
        }
    """
    total_size = 12812 # change according to desired dataset size; 12812 is the size of the smallest direction dataset; 50'000 is for the ones with >100000 examples; 1500 is for the run to compare to wmt models
    train_size = int(total_size * 0.8) 
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)

    print(train_size + test_size + val_size, total_size)
    assert train_size + test_size + val_size in range(total_size - 2, total_size + 2)

    for lang_pair in lang_pairs:
        lang1, lang2 = lang_pair.split("-")
        lang_pair_path = os.path.join(path_to_data, lang_pair.upper(), "tab")

        train_set = TranslationDataset(
                name='europarl',
                type='ht',
                src_lang=lang1,
                tgt_lang=lang2,
                examples=[],
            )
        test_set = TranslationDataset(
                name='europarl',
                type='ht',
                src_lang=lang1,
                tgt_lang=lang2,
                examples=[],
            )
        val_set = TranslationDataset(
                name='europarl',
                type='ht',
                src_lang=lang1,
                tgt_lang=lang2,
                examples=[],
            )
        
        counter = 0
        for file in os.listdir(lang_pair_path):
            with open(os.path.join(lang_pair_path, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    counter += 1
                    if train_set.num_examples < train_size:
                        example = TranslationExample(
                            src=lines[i].split('\t')[0].strip(),
                            tgt=lines[i].split('\t')[1].strip(),
                            docid=file.replace('.tab', ''),
                            sysid=file.replace('.tab', '')+'-'+str(i)
                        )
                        train_set.examples.append(example)
                    elif test_set.num_examples < test_size:
                        example = TranslationExample(
                            src=lines[i].split('\t')[0].strip(),
                            tgt=lines[i].split('\t')[1].strip(),
                            docid=file.replace('.tab', ''),
                            sysid=file.replace('.tab', '')+'-'+str(i)
                        )
                        test_set.examples.append(example)
                    elif val_set.num_examples < val_size:
                        example = TranslationExample(
                            src=lines[i].split('\t')[0].strip(),
                            tgt=lines[i].split('\t')[1].strip(),
                            docid=file.replace('.tab', ''),
                            sysid=file.replace('.tab', '')+'-'+str(i)
                        )
                        val_set.examples.append(example)
                    else:
                        break
                if train_set.num_examples == train_size and test_set.num_examples == test_size and val_set.num_examples == val_size:
                    break
                
        print(train_set.num_examples, test_set.num_examples, val_set.num_examples, train_set.num_examples + test_set.num_examples + val_set.num_examples, total_size)
        assert train_set.num_examples == train_size
        assert test_set.num_examples == test_size
        assert val_set.num_examples == val_size

        train_datasets.append(train_set)
        val_datasets.append(val_set)
        test_datasets.append(test_set)
        
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

