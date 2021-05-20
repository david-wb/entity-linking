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

    def test_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=zeshel_e2e_collate_fn)
        inputs = list(loader)[0]
        context_ids = inputs['context_ids']
        context_mask = inputs['context_mask']
        entity_ids = inputs['entity_ids']
        self.assertSequenceEqual(context_ids.shape, (2, 512))
        self.assertSequenceEqual(entity_ids.shape, (2, 1, 512))

    def test_biencoder_forward(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=zeshel_e2e_collate_fn)
        inputs = list(loader)[0]
        context_ids = inputs['context_ids']
        context_mask = inputs['context_mask']
        entity_ids = inputs['entity_ids']
        entity_mask = inputs['entity_mask']
        all_entities_mask = inputs['all_entities_mask']
        mention_starts = inputs['mention_starts']
        mention_ends = inputs['mention_ends']
        mention_middles = inputs['mention_middles']
        mention_bounds = inputs['mention_bounds']
        mention_bounds_mask = inputs['mention_bounds_mask']

        outputs = self.model(
            context_ids=context_ids,
            context_mask=context_mask,
            entity_ids=entity_ids,
            entity_mask=entity_mask,
            all_entities_mask=all_entities_mask,
            mention_starts=mention_starts,
            mention_ends=mention_ends,
            mention_middles=mention_middles,
            mention_bounds=mention_bounds,
            mention_bounds_mask=mention_bounds_mask
        )
        self.assertIsNotNone(outputs)

    def test_train_bi_encoder(self):
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for i, batch in enumerate(self.loader):
            self.model.zero_grad()
            loss = self.model.training_step(batch, i)
            loss.backward()
            optimizer.step()
