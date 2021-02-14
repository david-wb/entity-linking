import os
from unittest import TestCase

import torch

from src.bi_encoder import BiEncoder
from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestBiEncoder(TestCase):
    def setUp(self):
        self.model = BiEncoder()
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train')

    def test_len(self):
        input = self.dataset[0]
        out = self.model(**input)
        self.assertEqual(out[0].detach().numpy().shape[-1], 768)

    def test_train_bi_encoder(self):
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for example in self.dataset:
            self.model.zero_grad()
            me, ee, loss = self.model(**example)
            print(loss.item())
            loss.backward()
            optimizer.step()
