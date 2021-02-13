import copy
import json
import os
from typing import List, Dict, Any

from numpy.random import randint
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


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

    def _get_negative_sample(self, idx) -> Dict:
        ni = idx
        while ni == idx:
            ni = randint(0, self.__len__())
        return self.mentions[ni]

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

        text = ' '.join(left_words) + ' # ' + ' '.join(mention_words) + ' # ' + ' '.join(right_words)

        tokens = self.tokenizer(text=text, return_tensors='pt', truncation=True)

        return tokens

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> List[str]:
        title = mention['label_document']['title']
        text = mention['label_document']['text']

        tokens = self.tokenizer(text=title + ' | ' + text, return_tensors='pt', truncation=True)

        return tokens

    def __getitem__(self, idx):
        mention = copy.deepcopy(self.mentions[idx])
        negative_sample = self._get_negative_sample(idx)

        assert mention['text'] != negative_sample['text']

        mention_inputs = self._get_mention_tokens(mention)
        entity_inputs = self._get_entity_tokens(mention)
        negative_entity_inputs = self._get_entity_tokens(negative_sample)

        return {
            'mention_inputs': mention_inputs,
            'entity_inputs': entity_inputs,
            'negative_entity_inputs': negative_entity_inputs
        }

