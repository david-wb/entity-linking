import os
import sys
import shutil
from argparse import ArgumentParser
from uuid import uuid4

import torch
from bavard_ml_common.mlops.gcs import GCSClient
from loguru import logger
import tarfile

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.transform_zeshel import transform_zeshel
from src.zeshel_dataset import ZeshelDataset
import pytorch_lightning as pl


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="The directory path (local or GCS) to save the final trained pipeline instance to."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="test",
        help="The file path (local or GCS) to fetch the zeshel data from."
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


def train(work_dir: str, data_dir: str):
    model = BiEncoder(device=DEVICE)
    model.train()
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    trainset = ZeshelDataset(data_dir,
        split='train', tokenizer=tokenizer, device=DEVICE)

    valset = ZeshelDataset(data_dir,
        split='val', tokenizer=tokenizer, device=DEVICE)

    print('Validation examples:', len(valset))
    valset = [valset[i] for i in range(100)]
    trainloader = DataLoader(trainset, batch_size=4, num_workers=12)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, num_workers=12)

    trainer = pl.Trainer(gpus=-1, val_check_interval=100, accumulate_grad_batches=32, log_every_n_steps=1)
    trainer.fit(model, trainloader, valloader)


def main():
    """Train/fine tune a response generator on the EmpatheticDialogues dataset. Expects GPU to be available.
    """
    args = parse_cli_args()

    work_dir = os.path.join(os.getcwd(), str(uuid4()))
    logger.info(f"Deleting the work folder {work_dir}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # Get data
    zeshel_dir = os.path.join(work_dir, 'zeshel')
    if GCSClient.is_gcs_uri(args.data_file):
        data_filename = os.path.join(work_dir, 'zeshel.tar.bz2')
        logger.info(f"Downloading zeshel data.")
        GCSClient().download_blob_to_filename(args.data_file, data_filename)
    else:
        data_filename = args.data_file

    logger.info(f"Extracting zeshel data.")
    tar = tarfile.open(data_filename, "r:bz2")
    tar.extractall(work_dir)
    tar.close()

    logger.info(f"Transforming zeshel data.")
    zeshel_transformed_dir = os.path.join(work_dir, 'transformed_zeshel')
    transform_zeshel(zeshel_dir, zeshel_transformed_dir)

    # Train
    logger.info(f"Training model.")
    train(work_dir, zeshel_transformed_dir)

    # Save
    if GCSClient.is_gcs_uri(args.job_dir):
        logsdir = os.path.join(work_dir, 'lightning_logs')
        GCSClient().upload_dir(logsdir, args.job_dir)
        # Clean up the intermediate copy.

    # shutil.rmtree(args.output_dir)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
