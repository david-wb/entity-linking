import os
from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestBiEncoder(TestCase):
    def setUp(self):
        self.model = BiEncoder()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=self.tokenizer)
        self.loader = DataLoader(self.dataset, batch_size=1)

    def test_len(self):
        input = list(self.loader)[0]
        out = self.model(**input)
        self.assertEqual(out[0].detach().numpy().shape[-1], 128)

    def test_train_bi_encoder(self):
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for i, batch in enumerate(self.loader):
            self.model.zero_grad()
            loss = self.model.training_step(batch, i)
            loss.backward()
            optimizer.step()
