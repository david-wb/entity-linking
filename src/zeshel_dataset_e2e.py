import json
import os
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ZeshelDatasetE2E(Dataset):
    """Zero-Shot Entity Linking Dataset"""

    def __init__(self,
                 zeshel_home: str,
                 split: str,
                 tokenizer: PreTrainedTokenizer,
                 transform=None,
                 device='cpu'):
        """
        Args:
            zeshel_home (string): Path to folder containing the transformed Zeshel data.
            split (string): train, val, or test.
            context_size (int): Number of words to keep on the left and right of the mention.
        """
        self.zeshel_home = zeshel_home
        self.transform = transform
        self.device = device
        zeshel_file = os.path.join(zeshel_home, f'{split}_docs.json')
        self.tokenizer = tokenizer

        self.classification_token = '[CLS]'
        self.sep_token = '[SEP]'

        with open(zeshel_file) as f:
            self.docs: Dict[str, Any] = json.load(f)

        self.docs_list = list(self.docs.values())

    def __len__(self):
        return len(self.docs_list)

    def _get_entity_tokens(self, label_doc: Dict) -> Dict:
        title = label_doc['title'].lower()
        text = label_doc['text'].lower()

        title_tokens = self.tokenizer.tokenize(title)
        text_tokens = self.tokenizer.tokenize(text)
        tokens = title_tokens + ['[ENT]'] + text_tokens
        tokens = [self.classification_token] + tokens[:self.tokenizer.model_max_length - 2] + [self.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += [0] * len(padding)

        assert len(input_ids) == self.tokenizer.model_max_length

        inputs = {
            'ids': torch.LongTensor(input_ids),
            'mask': torch.LongTensor(attention_mask),
        }
        return inputs

    def __getitem__(self, idx):
        doc = self.docs_list[idx]

        text = doc['text']
        words = text.split()
        word_to_token_idx = {}
        doc_tokens = []

        for i, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            word_to_token_idx[i] = (len(doc_tokens), len(doc_tokens) + len(tokens) - 1)
            doc_tokens += tokens

        mention_bounds = []
        mention_entity_ids = []
        mention_entity_inputs = []
        for mention in doc['mentions']:
            start_word_i = mention['start_index']
            end_word_i = mention['end_index']

            start_token_i = word_to_token_idx[start_word_i][0]
            end_token_i = word_to_token_idx[end_word_i][1]

            if end_token_i > self.tokenizer.model_max_length - 1:
                break

            mention_bounds.append((start_token_i + 1, end_token_i + 1))  # Add 1 for CLS token
            mention_entity_ids.append(mention['label_document_id'])
            mention_entity_inputs.append(self._get_entity_tokens(mention['label_doc']))

        doc_tokens = [self.classification_token] + doc_tokens[:(self.tokenizer.model_max_length - 2)] + [self.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        attention_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(input_ids))

        input_ids += padding
        attention_mask += [0] * len(padding)

        starts = [0] * len(input_ids)
        ends = [0] * len(input_ids)
        middles = [0] * len(input_ids)

        for (s, e) in mention_bounds:
            starts[s] = 1
            ends[e] = 1
            for i in range(s + 1, e):
                middles[i] = 1

        assert len(input_ids) == self.tokenizer.model_max_length

        inputs = {
            'context_ids': input_ids,
            'context_attention_mask': attention_mask,
            'mention_bounds': mention_bounds,
            'mention_entity_ids': mention_entity_ids,
            'mention_entity_inputs': mention_entity_inputs,
            'starts': starts,
            'ends': ends,
            'middles': middles,
        }
        return inputs


def zeshel_e2e_collate_fn(batch):
    context_ids = torch.LongTensor([x['context_ids'] for x in batch])
    context_attention_mask = torch.LongTensor([x['context_attention_mask'] for x in batch])
    mention_starts = torch.LongTensor([x['starts'] for x in batch])
    mention_ends = torch.LongTensor([x['ends'] for x in batch])
    mention_middles = torch.LongTensor([x['middles'] for x in batch])
    mention_bounds = [x['mention_bounds'] for x in batch]
    mention_entity_inputs = [x['mention_entity_inputs'] for x in batch]

    return {
        'context_ids': context_ids,
        'context_attention_mask': context_attention_mask,
        'mention_starts': mention_starts,
        'mention_ends': mention_ends,
        'mention_middles': mention_middles,
        'mention_bounds': mention_bounds,
        'mention_entity_inputs': mention_entity_inputs
    }