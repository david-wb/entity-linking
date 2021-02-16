import os
from unittest import TestCase

from transformers import BertTokenizer
from transformers.models.auto.tokenization_auto import BertTokenizerFast

from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDataset(TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.dataset = ZeshelDataset(os.path.join(dir_path, 'data'), split='train', tokenizer=self.tokenizer)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_special_token(self):
        sample = self.dataset[0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(sample['mention_inputs']['input_ids'])
        print(input_tokens)
        self.assertEqual(input_tokens[0], '[CLS]')
        self.assertEqual(input_tokens[-1], '[PAD]')
