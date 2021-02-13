import copy
import json
import os
from numpy.random import randint
from typing import List, Dict, Any

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
        # left_tokens = self.tokenizer.tokenize(' '.join(left_words))
        # right_tokens = self.tokenizer.tokenize(' '.join(right_words))
        # mention_tokens = self.tokenizer.tokenize(' '.join(mention_words))
        #
        # tokens = ['[CLS]'] + left_tokens + ['[SEP]'] + right_tokens + ['[SEP]'] + mention_tokens
        # while len(tokens) < 2 * self.context_size + 12:
        #     tokens.append('[PAD]')
        #
        # assert len(tokens) == 2 * self.context_size + 12

        tokens = self.tokenizer(text=text, return_tensors='pt', truncation=True)

        return tokens

    def _get_entity_tokens(self, mention: Dict[str, Any]) -> List[str]:
        title = mention['label_document']['title']
        text = mention['label_document']['text']

        # title_tokens = self.tokenizer.tokenize(title)[:9]
        # text_tokens = self.tokenizer.tokenize(text)[:99]
        #
        # tokens = ['[CLS]'] + title_tokens + ['[SEP]'] + text_tokens  # max len is 110
        # while len(tokens) < 110:
        #     tokens.append('[PAD]')

        tokens = self.tokenizer(text=title + ' | ' + text, return_tensors='pt', truncation=True)

        #assert len(tokens) == 110
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

