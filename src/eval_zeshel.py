import sys
from argparse import ArgumentParser

import numpy as np
from loguru import logger
from sklearn.neighbors import NearestNeighbors


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


def evaluate(checkpoint_path: str, data_dir: str, batch_size: int):
    pass


def main():
    args = parse_cli_args()

    logger.info(f"Loading embeddings.")
    mentions = np.load(args.mention_embeddings, allow_pickle=True)
    entities = np.load(args.entity_embeddings, allow_pickle=True)
    mention_embeddings = mentions.item().get('embeddings')
    total_mentions = mention_embeddings.shape[0]
    entity_ids = entities.item().get('ids')
    total_entities = len(entity_ids)

    retrieval_rate_5 = compute_retrieval_rate(mentions, entities, k=5)
    print('Retrieval rate (k=5):', retrieval_rate_5, f', (expected rate for random = {5/total_entities})')

    retrieval_rate = compute_retrieval_rate(mentions, entities, k=10)
    print('Retrieval rate (k=10):', retrieval_rate, f', (expected rate for random = {10/total_entities})')

    retrieval_rate = compute_retrieval_rate(mentions, entities, k=20)
    print('Retrieval rate (k=20):', retrieval_rate, f', (expected rate for random = {20/total_entities})')

    retrieval_rate = compute_retrieval_rate(mentions, entities, k=50)
    print('Retrieval rate (k=50):', retrieval_rate, f', (expected rate for random = {50/total_entities})')


def compute_retrieval_rate(mentions, entities, k: int) -> float:
    mention_embeddings = mentions.item().get('embeddings')
    entity_embeddings = entities.item().get('embeddings')
    entity_ids = entities.item().get('ids')
    mention_entity_ids = mentions.item().get('entity_ids')

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(entity_embeddings)
    d, indices = nbrs.kneighbors(mention_embeddings)

    total_mentions = mention_embeddings.shape[0]
    n = 0
    for i, row in enumerate(indices):
        true_id = mention_entity_ids[i]
        if true_id in [entity_ids[i] for i in row]:
            n += 1
    return n / total_mentions


if __name__ == '__main__':
    main()