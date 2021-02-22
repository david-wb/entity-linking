import os
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.zeshel_dataset import ZeshelDataset


def train_zeshel(work_dir: str,
                 data_dir: str,
                 batch_size: int,
                 val_check_interval: int,
                 limit_train_batches: Optional[int] = None,
                 max_epochs: int = 1):
    model = BiEncoder()
    model.train()
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    trainset = ZeshelDataset(data_dir,
                             split='train', tokenizer=tokenizer, device=DEVICE)

    valset = ZeshelDataset(data_dir,
                           split='val', tokenizer=tokenizer, device=DEVICE)

    print('Validation examples:', len(valset))
    valset = [valset[i] for i in range(100)]
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=12)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=12)

    accumulate_grad_batches = max(1, 128 // batch_size)
    wandb_logger = WandbLogger(project='entity-linker')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        verbose=True,
        filepath=os.path.join(work_dir, f'checkpoints/entity_linker_{datetime.now().strftime("%m%d%H%M%S")}')
    )
    trainer = pl.Trainer(
        gpus=-1 if DEVICE != 'cpu' else 0,
        logger=[wandb_logger],
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=1,
        limit_train_batches=limit_train_batches if limit_train_batches else 1.0,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs)
    trainer.fit(model, trainloader, valloader)
