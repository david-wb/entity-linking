import os
from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from src.bi_encoder_e2e import BiEncoderE2E
from src.zeshel_dataset_e2e import ZeshelDatasetE2E, zeshel_e2e_collate_fn

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestBiEncoderE2E(TestCase):
    def setUp(self):
        self.model = BiEncoderE2E()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.dataset = ZeshelDatasetE2E(os.path.join(dir_path, 'data'), split='tiny', tokenizer=self.tokenizer)
        self.loader = DataLoader(self.dataset, batch_size=2, collate_fn=zeshel_e2e_collate_fn)

    def test_len(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=zeshel_e2e_collate_fn)
        inputs = list(loader)[0]
        out = self.model(inputs)
        self.assertSequenceEqual(out.detach().numpy().shape, (2, 512, 3))

    def test_train_bi_encoder(self):
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for i, batch in enumerate(self.loader):
            self.model.zero_grad()
            loss = self.model.training_step(batch, i)
            loss.backward()
            optimizer.step()
