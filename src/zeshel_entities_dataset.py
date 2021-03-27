import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.enums import BaseModelType


class ZeshelEntitiesDataset(Dataset):
    def __init__(self,
                 zeshel_home: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 base_model_type: str):
        self.base_model_type = base_model_type
        self.zeshel_home = zeshel_home
        entities_file = os.path.join(zeshel_home, f'entities_{split}.json')
        self.tokenizer = tokenizer

        if base_model_type == BaseModelType.BERT_BASE.name:
            self.classification_token = '[CLS]'
            self.sep_token = '[SEP]'
        else:
            self.classification_token = '<s>'
            self.sep_token = '</s>'

        with open(entities_file) as f:
            self.entities: List[tuple] = list(json.load(f).items())

    def __len__(self):
        return len(self.entities)

    def _get_entity_tokens(self, entity: Dict[str, Any]) -> Dict:
        title = entity['title'].lower()
        text = entity['text'].lower()

        title_tokens = self.tokenizer.tokenize(title)
        text_tokens = self.tokenizer.tokenize(text)
        tokens = title_tokens + ['|'] + text_tokens
        tokens = [self.classification_token] + tokens[:self.tokenizer.model_max_length - 2] + [self.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += padding

        assert len(input_ids) <= 512

        inputs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
        }
        return inputs

    def __getitem__(self, idx):
        doc_id, entity = self.entities[idx]
        entity_inputs = self._get_entity_tokens(entity)
        return doc_id, entity_inputs
