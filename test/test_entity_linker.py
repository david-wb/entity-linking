import os
from unittest import TestCase

import torch

from src.entity_linker_model import EntityLinker
from src.zeshel_dataset import ZeshelDataset
dir_path = os.path.dirname(os.path.realpath(__file__))


class TestEntityLinker(TestCase):
    def setUp(self):
        self.model = EntityLinker()
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train')

    def test_len(self):
        input = self.dataset[0]
        out = self.model(**input)
        self.assertEqual(out[0].detach().numpy().shape[-1], 768)
