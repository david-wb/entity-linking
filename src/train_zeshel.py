import os
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.enums import BaseModelType
from src.tokenization import get_tokenizer
from src.zeshel_dataset import ZeshelDataset


def train_zeshel(work_dir: str,
                 data_dir: str,
                 batch_size: int,
                 val_check_interval: int,
                 limit_train_batches: Optional[int] = None,
                 max_epochs: int = 1,
                 base_model_type: str = BaseModelType.BERT_BASE.name):
    model = BiEncoder(base_model_type=base_model_type)
    model.train()
    model.to(DEVICE)

    tokenizer = get_tokenizer(base_model_type)
    trainset = ZeshelDataset(data_dir, split='train', tokenizer=tokenizer, device=DEVICE,
                             base_model_type=base_model_type)
    valset = ZeshelDataset(data_dir, split='val', tokenizer=tokenizer, device=DEVICE,
                           base_model_type=base_model_type)
    print('Validation examples:', len(valset))
    valset = [valset[i] for i in range(100)]
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=12, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=12, shuffle=True)

    accumulate_grad_batches = max(1, 128 // batch_size)
    wandb_logger = WandbLogger(project='entity-linker')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        verbose=True,
        dirpath=os.path.join(work_dir, f'checkpoints'),
        filename='{epoch}-{val_loss:.3f}' + f'_{base_model_type}_{datetime.now().strftime("%m_%d_%H%M_%S")}'
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
