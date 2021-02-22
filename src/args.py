from argparse import ArgumentParser, Namespace
from typing import List


def parse_zeshel_train_args(args: List[str]) -> Namespace:
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
    parsed_args = parser.parse_args(args)
    return parsed_args
