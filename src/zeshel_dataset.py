import copy
import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel


class ZeshelDataset(Dataset):
    """Zero-Shot Entity Linking Dataset"""

    def __init__(self, zeshel_home: str, split: str, context_size=10, transform=None):
        """
        Args:
            zeshel_home (string): Path to folder containing the transformed Zeshel data.
            split (string): train, val, or test.
            context_size (int): Number of words to keep on the left and right of the mention.
        """
        self.zeshel_home = zeshel_home
        self.transform = transform
        self.context_size = context_size

        zeshel_file = os.path.join(zeshel_home, f'mentions_{split}.json')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        with open(zeshel_file) as f:
            self.mentions: List[Dict] = list(json.load(f).values())

    def __len__(self):
        return len(self.mentions)

    def _get_mention_tokens(self, mention: Dict[str, Any]):
        start_i = mention['start_index']
        end_i = mention['end_index']
        mention_text = mention['text']
        words = mention['source_document']['text'].split()
        window_start = max(0, start_i - self.context_size)

        mention_words = mention_text.split()
        mention_words = mention_words[:10]
        left_words = words[window_start:start_i]
        right_words = words[end_i + 1: end_i + self.context_size]

        window = left_words + ['[SEP]'] + mention_words + ['[SEP]'] + right_words
        while len(window) < 2 * self.context_size + 12:
            window.append('[PAD]')

        assert len(window) == 2 * self.context_size + 12

        window_text = ' '.join(window)
        mention_tokens = self.tokenizer.tokenize(window_text)
        return mention_tokens

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> List[str]:
        title = mention['label_document']['title']
        text = mention['label_document']['text']

        title_words = title.split()[:9]
        while len(title_words) < 9:
            title_words.append('[PAD]')

        words = title_words + ['[SEP]'] + text.split()[:100]  # max len is 110
        while len(words) < 110:
            words.append('[PAD]')

        assert len(words) == 110

        entity_text = ' '.join(words).strip()
        entity_tokens = self.tokenizer.tokenize(entity_text)
        return entity_tokens

    def __getitem__(self, idx):
        mention = copy.deepcopy(self.mentions[idx])
        mention['mention_tokens'] = self._get_mention_tokens(mention)
        mention['entity_tokens'] = self._get_entity_tokens(mention)

        print(mention['mention_tokens'])
        print(mention['entity_tokens'])

        return mention

