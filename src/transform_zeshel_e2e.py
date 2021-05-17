import json
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict
from itertools import islice


def add_mentions(input_dir: str, split: str, all_docs: Dict):
    with open(os.path.join(input_dir, f'mentions/{split}.json')) as f:
        lines = f.readlines()
    mentions = [json.loads(line) for line in lines]

    for men in mentions:
        context_document_id = men['context_document_id']
        men['label_doc'] = {
            'title': all_docs[men['label_document_id']]['title'],
            'text': all_docs[men['label_document_id']]['text']
        }
        doc = all_docs[context_document_id]
        if 'mentions' not in doc:
            doc['mentions'] = []
        doc['mentions'].append(men)
        doc['split'] = split


def transform_zeshel_e2e(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_docs = {}
    documents_dir = os.path.join(input_dir, 'documents/')
    for filename in os.listdir(documents_dir):
        if filename.endswith(".json"):
            corpus = os.path.splitext(os.path.basename(filename))[0]
            with open(os.path.join(documents_dir, filename)) as f:
                corpus_lines = f.readlines()

            docs = [json.loads(line) for line in corpus_lines]
            for doc in docs:
                doc['corpus'] = corpus
            all_docs.update({doc['document_id']: doc for doc in docs})

    for split in ['train', 'val', 'test']:
        add_mentions(input_dir, split, all_docs)

    # split docs
    splits = {'train': {}, 'val': {}, 'test': {}, 'tiny': {}}
    for doc_id, doc in all_docs.items():
        if 'split' not in doc:
            doc['split'] = 'test'

        splits[doc['split']][doc_id] = doc

    # add tiny split
    for doc_id, doc in islice(splits['train'].items(), 10):
        splits['tiny'][doc_id] = doc

    for split, docs in splits.items():
        docs_path = os.path.join(output_dir, f'{split}_docs.json')
        with open(docs_path, 'w') as f:
            json.dump(docs, f, indent=2)

    all_documents_path = os.path.join(output_dir, f'all_documents.json')
    with open(all_documents_path, 'w') as f:
        json.dump(all_docs, f, indent=2)

    print('Total entities:', len(all_docs))


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
        default='./zeshel_transformed_e2e',
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == '__main__':
    args = parse_cli_args()
    transform_zeshel_e2e(args.input_dir, args.output_dir)
