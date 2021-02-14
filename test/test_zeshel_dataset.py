import os
from unittest import TestCase

from src.zeshel_dataset import ZeshelDataset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestZeshelDataset(TestCase):
    def setUp(self):
        self.dataset = ZeshelDataset(os.path.join(dgitir_path, 'data'), split='train')

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)
