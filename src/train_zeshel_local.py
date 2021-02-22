import os
import sys

from loguru import logger

from src.args import parse_zeshel_train_args
from src.train_zeshel import train_zeshel


def main():
    """Train/fine tune a response generator on the EmpatheticDialogues dataset. Expects GPU to be available.
    """
    args = parse_zeshel_train_args(sys.argv[1:])

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
                 max_epochs=args.max_epochs)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
