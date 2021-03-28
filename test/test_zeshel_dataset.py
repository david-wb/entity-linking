import os
from unittest import TestCase

from torch.utils.data import DataLoader

from src.enums import BaseModelType
from src.tokenization import get_tokenizer
from src.zeshel_dataset import ZeshelDataset
from src.zeshel_entities_dataset import ZeshelEntitiesDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDataset(TestCase):
    def setUp(self):
        base_model_type = BaseModelType.BERT_BASE.name
        self.tokenizer = get_tokenizer(base_model_type)
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=self.tokenizer,
                                     base_model_type=base_model_type)
        self.entities_dataset = ZeshelEntitiesDataset(os.path.join(dir_path, 'data'), split='train',
                                                      tokenizer=self.tokenizer, base_model_type=base_model_type)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_special_token(self):
        sample = self.dataset[0]
        mention_input_tokens = self.tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        entity_input_tokens = self.tokenizer.convert_ids_to_tokens(sample['entity_inputs']['input_ids'])
        self.assertEqual(mention_input_tokens[0], '[CLS]')
        self.assertEqual(mention_input_tokens[-1], '[PAD]')
        self.assertEqual(entity_input_tokens[0], '[CLS]')
        self.assertEqual(entity_input_tokens[-1], '[SEP]')

    def test_special_token_roberta(self):
        base_model_type = BaseModelType.ROBERTA_BASE.name
        tokenizer = get_tokenizer(base_model_type)
        dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=tokenizer,
                                base_model_type=base_model_type)
        sample = dataset[0]
        mention_input_tokens = tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        entity_input_tokens = tokenizer.convert_ids_to_tokens(sample['entity_inputs']['input_ids'])
        print(mention_input_tokens)
        self.assertEqual(mention_input_tokens[0], '<s>')
        self.assertEqual(mention_input_tokens[-1], '<pad>')
        self.assertEqual(entity_input_tokens[0], '<s>')
        self.assertEqual(entity_input_tokens[-1], '</s>')

    def test_special_token_declutr(self):
        base_model_type = BaseModelType.DECLUTR_BASE.name
        tokenizer = get_tokenizer(base_model_type)
        dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=tokenizer,
                                base_model_type=base_model_type)
        sample = dataset[0]
        mention_input_tokens = tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        entity_input_tokens = tokenizer.convert_ids_to_tokens(sample['entity_inputs']['input_ids'])
        self.assertEqual(mention_input_tokens[0], '<s>')
        self.assertEqual(mention_input_tokens[-1], '<pad>')
        self.assertEqual(entity_input_tokens[0], '<s>')
        self.assertEqual(entity_input_tokens[-1], '</s>')

    def test_loader(self):
        loader = DataLoader(self.dataset, batch_size=2)
        self.assertEqual(len(loader), 2)

    def test_entities_dataset(self):
        loader = DataLoader(self.entities_dataset, batch_size=2)
        for ids, inputs in loader:
            print(ids)
        self.assertEqual(len(loader), 2)
