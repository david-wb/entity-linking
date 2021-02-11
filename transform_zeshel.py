import json
import os
from typing import Dict

dir_path = os.path.dirname(os.path.realpath(__file__))


def transform_mentions(split: str, corpus_dict: Dict):
    with open(os.path.join(dir_path, f'zeshel/mentions/{split}.json')) as f:
        lines = f.readlines()
    mentions = [json.loads(line) for line in lines]
    for men in mentions:
        men['source_document'] = corpus_dict[men['corpus']][men['context_document_id']]
    mentions_dict = {m['mention_id']: m for m in mentions}

    mentions_file = os.path.join(dir_path, f'zeshel_transformed/mentions_{split}.json')
    if not os.path.exists(os.path.dirname(mentions_file)):
        os.makedirs(os.path.dirname(mentions_file))
    with open(mentions_file, 'w') as f:
        json.dump(mentions_dict, f, indent=2)


def transform_zeshel():
    corpus_dict = {}
    documents_dir = os.path.join(dir_path, 'zeshel/documents/')
    for filename in os.listdir(documents_dir):
        if filename.endswith(".json"):
            corpus = os.path.splitext(os.path.basename(filename))[0]
            with open(os.path.join(documents_dir, filename)) as f:
                corpus_lines = f.readlines()

            docs = [json.loads(line) for line in corpus_lines]
            corpus_dict[corpus] = {doc['document_id']: doc for doc in docs}

    for split in ['train', 'val', 'test']:
        transform_mentions(split, corpus_dict)


if __name__ == '__main__':
    transform_zeshel()
