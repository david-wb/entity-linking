import os
from unittest import TestCase

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.zeshel_dataset_declutr import ZeshelDatasetDeCLUTR
from src.zeshel_entities_dataset_declutr import ZeshelEntitiesDatasetDeCLUTR

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDatasetDeCLUTR(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained('johngiorgi/declutr-base', do_lower_case=True)
        self.dataset = ZeshelDatasetDeCLUTR(os.path.join(dir_path, 'data'), split='train', tokenizer=self.tokenizer)
        self.entities_dataset = ZeshelEntitiesDatasetDeCLUTR(os.path.join(dir_path, 'data'), split='train',
                                                      tokenizer=self.tokenizer)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_special_token(self):
        sample = self.dataset[0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        print(input_tokens)

    def test_loader(self):
        loader = DataLoader(self.dataset, batch_size=2)
        self.assertEqual(len(loader), 2)

    def test_entities_dataset(self):
        loader = DataLoader(self.entities_dataset, batch_size=2)
        for ids, inputs in loader:
            print(ids)
        self.assertEqual(len(loader), 2)
