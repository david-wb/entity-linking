import os
from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from src.enums import BaseModelType
from src.tokenization import get_tokenizer
from src.zeshel_dataset_e2e import ZeshelDatasetE2E, zeshel_e2e_collate_fn

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDataset(TestCase):
    def setUp(self):
        base_model_type = BaseModelType.BERT_BASE.name
        self.tokenizer = get_tokenizer(base_model_type)
        self.dataset = ZeshelDatasetE2E(os.path.join(dir_path, 'data'), split='tiny', tokenizer=self.tokenizer)

    def test_len(self):
        self.assertEqual(len(self.dataset), 10)

    def test_mention_bounds(self):
        sample = self.dataset[1]
        self.assertEqual(len(sample['mention_bounds']), 1)

    def test_loader(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=zeshel_e2e_collate_fn)

        for batch in loader:
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(k, v.shape)
                else:
                    print(k, v)

        self.assertEqual(len(loader), 5)
