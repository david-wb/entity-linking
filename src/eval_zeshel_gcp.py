import os
import sys
import tarfile
from argparse import ArgumentParser
from uuid import uuid4

from bavard_ml_common.mlops.gcs import GCSClient
from loguru import logger

from src.compute_embeddings import embedd_mentions, embedd_entities
from src.eval_zeshel import eval_zeshel
from src.transform_zeshel import transform_zeshel


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="The directory path (local or GCS) to the final trained model."
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
        "--base-model-type",
        type=str,
        choices=['BERT_BASE', 'DECLUTR_BASE'],
        required=True
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def main():
    args = parse_cli_args()

    work_dir = os.path.join(os.getcwd(), str(uuid4()))
    os.makedirs(work_dir)
    os.chdir(work_dir)

    # Get data
    zeshel_dir = os.path.join(work_dir, 'zeshel')
    if GCSClient.is_gcs_uri(args.data_file):
        data_filename = os.path.join(work_dir, 'zeshel.tar.bz2')
        logger.info(f"Downloading zeshel data.")
        GCSClient().download_blob_to_filename(args.data_file, data_filename)
    else:
        data_filename = args.data_file

    if GCSClient.is_gcs_uri(args.checkpoint_path):
        checkpoint_path = os.path.join(work_dir, 'checkpoint.ckpt')
        logger.info(f"Downloading checkpoint file.")
        GCSClient().download_blob_to_filename(args.checkpoint_path, checkpoint_path)
    else:
        checkpoint_path = args.checkpoint_path

    logger.info(f"Extracting zeshel data.")
    tar = tarfile.open(data_filename, "r:bz2")
    tar.extractall(work_dir)
    tar.close()

    logger.info(f"Transforming zeshel data.")
    zeshel_transformed_dir = os.path.join(work_dir, 'transformed_zeshel')
    transform_zeshel(zeshel_dir, zeshel_transformed_dir)

    # Compute embeddings.
    logger.info(f"Computing mention embeddings.")
    embedd_mentions(
        checkpoint_path=checkpoint_path,
        data_dir=zeshel_transformed_dir,
        batch_size=4,
        base_model_type=args.base_model_type)

    logger.info(f"Computing entity embeddings.")
    embedd_entities(
        checkpoint_path=checkpoint_path,
        data_dir=zeshel_transformed_dir,
        batch_size=4,
        base_model_type=args.base_model_type)

    logger.info(f"Evaluating.")
    eval_zeshel(
        mention_embeddings=os.path.join(work_dir, 'zeshel_mention_embeddings_val.npy'),
        entity_embeddings=os.path.join(work_dir, 'zeshel_entity_embeddings_val.npy'),
    )

    # Save
    # TODO


if __name__ == '__main__':
    main()
