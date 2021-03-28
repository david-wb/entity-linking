import os
import sys
from argparse import ArgumentParser

from loguru import logger

from src.train_zeshel import train_zeshel


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
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--base-model-type",
        type=str,
        choices=['BERT_BASE', 'ROBERTA_BASE', 'DECLUTR_BASE'],
        required=True
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def main():
    args = parse_cli_args()

    work_dir = os.getcwd()

    # Get data
    zeshel_transformed_dir = os.path.join(work_dir, 'transformed_zeshel')

    # Train
    logger.info(f"Training model.")
    train_zeshel(work_dir,
                 zeshel_transformed_dir,
                 batch_size=args.batch_size,
                 val_check_interval=args.val_check_interval,
                 limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
                 max_epochs=args.max_epochs,
                 base_model_type=args.base_model_type)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
