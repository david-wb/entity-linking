import json
import os
from typing import List, Dict, Any

import torch
from numpy.random import randint
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import MENTION_START_TAG, MENTION_END_TAG


class ZeshelEntitiesDataset(Dataset):
    def __init__(self,
                 zeshel_home: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer):
        self.zeshel_home = zeshel_home
        entities_file = os.path.join(zeshel_home, f'entities_{split}.json')
        self.tokenizer = tokenizer

        with open(entities_file) as f:
            self.entities: List[Dict] = list(json.load(f).values())

    def __len__(self):
        return len(self.entities)

    def _get_entity_tokens(self, entity: Dict[str, Any]) -> Dict:
        title = entity['title'].lower()
        text = entity['text'].lower()

        title_tokens = self.tokenizer.tokenize(title)
        text_tokens = self.tokenizer.tokenize(text)
        tokens = title_tokens + ['unused2'] + text_tokens
        tokens = tokens[:self.tokenizer.model_max_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        padding = [0] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        inputs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'token_type_ids': torch.LongTensor(token_type_ids),
        }
        return inputs

    def __getitem__(self, idx):
        entity = self.entities[idx]
        entity_inputs = self._get_entity_tokens(entity)
        return {
            'entity_inputs': entity_inputs,
        }

