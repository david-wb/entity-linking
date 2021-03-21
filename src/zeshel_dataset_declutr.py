import json
import os
from typing import List, Dict, Any

import torch
from numpy.random import randint
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ZeshelDatasetDeCLUTR(Dataset):
    def __init__(self,
                 zeshel_home: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 context_size=32,
                 transform=None,
                 device='cpu'):
        self.zeshel_home = zeshel_home
        self.transform = transform
        self.context_size = context_size
        self.device = device
        zeshel_file = os.path.join(zeshel_home, f'mentions_{split}.json')
        self.tokenizer = tokenizer

        with open(zeshel_file) as f:
            self.mentions: List[Dict] = list(json.load(f).values())

    def __len__(self):
        return len(self.mentions)

    def _get_negative_sample(self, idx) -> Dict:
        ni = idx
        while ni == idx:
            ni = randint(0, self.__len__())
        return self.mentions[ni]

    def _get_mention_tokens(self, mention: Dict[str, Any]):
        start_i = mention['start_index']
        end_i = mention['end_index']
        mention_text = mention['text'].lower()
        words = mention['source_document']['text'].lower().split()

        mention_tokens = ['|'] + self.tokenizer.tokenize(mention_text) + ['|']
        left_tokens = self.tokenizer.tokenize(' '.join(words[:start_i]))
        right_tokens = self.tokenizer.tokenize(' '.join(words[end_i + 1:]))

        keep_left = (self.context_size - len(mention_tokens)) // 2
        keep_right = (self.context_size - keep_left - len(mention_tokens))
        ctx_tokens = left_tokens[-keep_left:] + mention_tokens + right_tokens[:keep_right]
        ctx_tokens = ctx_tokens[:self.tokenizer.model_max_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        padding = [0] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        assert len(input_ids) <= 512

        inputs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
        }
        return inputs

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> Dict:
        title = mention['label_document']['title'].lower()
        text = mention['label_document']['text'].lower()

        title_tokens = self.tokenizer.tokenize(title)
        text_tokens = self.tokenizer.tokenize(text)
        tokens = title_tokens + ['|'] + text_tokens
        tokens = tokens[:self.tokenizer.model_max_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        padding = [0] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        assert len(input_ids) <= 512

        inputs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
        }
        return inputs

    def __getitem__(self, idx):
        mention = self.mentions[idx]

        mention_inputs = self._get_mention_tokens(mention)
        entity_inputs = self._get_entity_tokens(mention)

        return {
            'entity_document_ids': mention['label_document_id'],
            'mention_document_ids': mention['context_document_id'],
            'mention_inputs': mention_inputs,
            'entity_inputs': entity_inputs,
        }

