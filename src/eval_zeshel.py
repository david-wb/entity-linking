import sys
from argparse import ArgumentParser

import numpy as np
from loguru import logger


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--mention-embeddings",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--entity-embeddings",
        type=str,
        required=True,
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def eval_zeshel(mention_embeddings: str, entity_embeddings: str):
    logger.info(f"Loading embeddings.")
    mentions = np.load(mention_embeddings, allow_pickle=True)
    entities = np.load(entity_embeddings, allow_pickle=True)
    entity_ids = entities.item().get('ids')
    total_entities = len(entity_ids)

    for k in [1, 4, 8, 16, 32, 64]:
        retrieval_rate = compute_retrieval_rate(mentions, entities, k=k)
        print(f'Retrieval rate (k={k}):', retrieval_rate, f', (expected rate for random = {k/total_entities})')


def compute_retrieval_rate(mentions, entities, k: int) -> float:
    mention_embeddings = mentions.item().get('embeddings')
    entity_embeddings = entities.item().get('embeddings')
    entity_ids = entities.item().get('ids')
    mention_entity_ids = mentions.item().get('entity_ids')

    scores = np.matmul(mention_embeddings, entity_embeddings.T)

    total_mentions = scores.shape[0]
    n = 0
    for i, row in enumerate(scores):
        indices = np.argsort(row)[::-1][:k]
        true_id = mention_entity_ids[i]
        if true_id in [entity_ids[i] for i in indices]:
            n += 1
    return n / total_mentions


def main():
    args = parse_cli_args()
    eval_zeshel(mention_embeddings=args.mention_embeddings,
                entity_embeddings=args.entity_embeddings)


if __name__ == '__main__':
    main()
