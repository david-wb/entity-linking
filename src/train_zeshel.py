import os
import shutil
import sys
import tarfile
from argparse import ArgumentParser
from uuid import uuid4

import pytorch_lightning as pl
import torch
from bavard_ml_common.mlops.gcs import GCSClient
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.transform_zeshel import transform_zeshel
from src.zeshel_dataset import ZeshelDataset


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="test",
        help="The file path (local or GCS) to zeshel.tar.bz2."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def validate(model, valloader):
    with torch.no_grad():
        total_loss = 0
        for batch in valloader:
            me, ee, loss = model(**batch)
            total_loss += loss.item()
        print('Validation loss', total_loss)


def train(work_dir: str, data_dir: str, batch_size: int):
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

    accumulate_grad_batches = min(1, 128 // batch_size)
    wandb_logger = WandbLogger(project='entity-linker')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        verbose=True,
        dirpath=os.path.join(work_dir, 'checkpoints'))
    trainer = pl.Trainer(
        gpus=-1 if DEVICE != 'cpu' else 0,
        logger=[wandb_logger],
        val_check_interval=1,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, valloader)


def main():
    """Train/fine tune a response generator on the EmpatheticDialogues dataset. Expects GPU to be available.
    """
    args = parse_cli_args()

    work_dir = os.getcwd()

    # Get data
    zeshel_transformed_dir = os.path.join(work_dir, 'transformed_zeshel')

    # Train
    logger.info(f"Training model.")
    train(work_dir, zeshel_transformed_dir, batch_size=args.batch_size)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
