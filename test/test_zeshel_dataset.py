import os
from unittest import TestCase

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.zeshel_dataset import ZeshelDataset
from src.zeshel_entities_dataset import ZeshelEntitiesDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDataset(TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=self.tokenizer)
        self.entities_dataset = ZeshelEntitiesDataset(os.path.join(dir_path, 'data'), split='train',
                                                      tokenizer=self.tokenizer)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_special_token(self):
        sample = self.dataset[0]
        mention_input_tokens = self.tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        entity_input_tokens = self.tokenizer.convert_ids_to_tokens(sample['entity_inputs']['input_ids'])
        print(entity_input_tokens)
        self.assertEqual(mention_input_tokens[0], '[CLS]')
        self.assertEqual(mention_input_tokens[-1], '[PAD]')

    def test_loader(self):
        loader = DataLoader(self.dataset, batch_size=2)
        self.assertEqual(len(loader), 2)

    def test_entities_dataset(self):
        loader = DataLoader(self.entities_dataset, batch_size=2)
        for ids, inputs in loader:
            print(ids)
        self.assertEqual(len(loader), 2)
