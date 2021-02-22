import json
import os
import sys
from argparse import ArgumentParser
from typing import Dict


def transform_mentions(input_dir: str, output_dir: str, split: str, corpus_dict: Dict):
    with open(os.path.join(input_dir, f'mentions/{split}.json')) as f:
        lines = f.readlines()
    mentions = [json.loads(line) for line in lines]
    for men in mentions:
        men['source_document'] = corpus_dict[men['corpus']][men['context_document_id']]
        men['label_document'] = corpus_dict[men['corpus']][men['label_document_id']]

    mentions_dict = {m['mention_id']: m for m in mentions}

    mentions_file = os.path.join(output_dir, f'mentions_{split}.json')
    if not os.path.exists(os.path.dirname(mentions_file)):
        os.makedirs(os.path.dirname(mentions_file))
    with open(mentions_file, 'w') as f:
        json.dump(mentions_dict, f, indent=2)

    return mentions_dict


def combine_entities(mentions: Dict, output_dir: str, split: str):
    entities = {}
    for men in mentions.values():
        ent_doc = men['label_document']
        entities[ent_doc['document_id']] = ent_doc

    entities_file = os.path.join(output_dir, f'entities_{split}.json')
    with open(entities_file, 'w') as f:
        json.dump(entities, f, indent=2)


def transform_zeshel(input_dir: str, output_dir: str):
    corpus_dict = {}
    documents_dir = os.path.join(input_dir, 'documents/')
    for filename in os.listdir(documents_dir):
        if filename.endswith(".json"):
            corpus = os.path.splitext(os.path.basename(filename))[0]
            with open(os.path.join(documents_dir, filename)) as f:
                corpus_lines = f.readlines()

            docs = [json.loads(line) for line in corpus_lines]
            corpus_dict[corpus] = {doc['document_id']: doc for doc in docs}

    for split in ['train', 'val', 'test']:
        mentions = transform_mentions(input_dir, output_dir, split, corpus_dict)
        combine_entities(mentions, output_dir, split)

        # Create tiny split for development
        if split == 'train':
            mentions_tiny = dict(list(mentions.items())[:100])
            mentions_tiny_file = os.path.join(output_dir, f'mentions_tiny.json')
            with open(mentions_tiny_file, 'w') as f:
                json.dump(mentions_tiny, f, indent=2)
                combine_entities(mentions_tiny, output_dir, 'tiny')


def parse_cli_args():
    args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='./transformed_zeshel',
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == '__main__':
    args = parse_cli_args()
    transform_zeshel(args.input_dir, args.output_dir)
