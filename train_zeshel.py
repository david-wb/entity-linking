import os

import torch
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.zeshel_dataset import ZeshelDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))


def validate(model, valloader):
    with torch.no_grad():
        total_loss = 0
        for batch in valloader:
            me, ee, loss = model(**batch)
            total_loss += loss.item()
        print('Validation loss', total_loss)


def main():
    # torch.cuda.empty_cache()
    # torch.multiprocessing.set_start_method('spawn')
    print('Cuda is available:', torch.cuda.is_available())
    device = 'cpu'  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BiEncoder(device=device)
    model.train()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    trainset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='train', tokenizer=tokenizer, device=device)
    valset = ZeshelDataset(os.path.join(dir_path, 'zeshel_transformed'), split='val', tokenizer=tokenizer, device=device)
    print('Validation examples:', len(valset))
    valset = [valset[i] for i in range(100)]
    trainloader = DataLoader(trainset, batch_size=4, num_workers=12)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, num_workers=12)

    trainer = pl.Trainer(val_check_interval=100, accumulate_grad_batches=5, log_every_n_steps=1)
    trainer.fit(model, trainloader, valloader)


if __name__ == '__main__':
    main()