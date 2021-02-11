import json
import os

import torch
from torch.utils.data import Dataset


class ZeshelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, zeshel_home: str, split: str, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.zeshel_home = zeshel_home
        self.transform = transform

        zeshel_file = os.path.join(zeshel_home, f'mentions_{split}.json')
        with open(zeshel_file) as f:
            self.mentions = list(json.load(f).values())

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx]
