import os
import shutil
import sys
import tarfile
from argparse import ArgumentParser
from uuid import uuid4

from bavard_ml_common.mlops.gcs import GCSClient
from loguru import logger

from src.train_zeshel import train_zeshel
from src.transform_zeshel import transform_zeshel


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="The directory path (local or GCS) to save the final trained model."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="test",
        help="The file path (local or GCS) to zeshel.tar.bz2."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
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
    parsed_args = parser.parse_args(args)
    return parsed_args


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
    train_zeshel(work_dir, zeshel_transformed_dir,
                 batch_size=args.batch_size,
                 val_check_interval=args.val_check_interval,
                 limit_train_batches=args.limit_train_batches,
                 max_epochs=args.max_epochs)
    logger.info(f"Training complete.")
    logger.info(f"Uploading checkpoints dir to GCS.")

    # Save
    if GCSClient.is_gcs_uri(args.job_dir):
        checkpoints_dir = os.path.join(work_dir, 'checkpoints')
        GCSClient().upload_dir(checkpoints_dir, args.job_dir)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
