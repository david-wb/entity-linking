import sys
from argparse import ArgumentParser

import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.zeshel_entities_dataset import ZeshelEntitiesDataset


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def embedd_entities(checkpoint_path: str, data_dir: str, batch_size: int):
    model = BiEncoder()
    model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    entities_dataset = ZeshelEntitiesDataset(data_dir, split='val', tokenizer=tokenizer)
    entities_loader = torch.utils.data.DataLoader(entities_dataset, batch_size=batch_size, num_workers=12)

    for ents_batch in entities_loader:
        entity_embeddings = model.entity_embedder(ents_batch)
        print(entity_embeddings)
        exit(0)


def main():
    args = parse_cli_args()

    # Train
    logger.info(f"Embeddings entities.")
    embedd_entities(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size)


if __name__ == '__main__':
    main()
