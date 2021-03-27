import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer

from src.bi_encoder import BiEncoder
from src.config import DEVICE
from src.enums import BaseModelType
from src.tokenization import get_tokenizer
from src.zeshel_dataset import ZeshelDataset
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
        "--data-dir",
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--base-model-type",
        type=str,
        choices=['BERT_BASE', 'ROBERTA_BASE', 'DECLUTR_BASE'],
        required=True
    )
    parsed_args = parser.parse_args(args)
    return parsed_args


def embedd_entities(checkpoint_path: str, data_dir: str, batch_size: int, base_model_type: str, split: str):
    model = BiEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device('cpu'),
        base_model_type=base_model_type
    )
    model.eval()
    model.to(DEVICE)

    if base_model_type == BaseModelType.BERT_BASE.name:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif base_model_type == BaseModelType.DECLUTR_BASE.name:
        tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
    else:
        raise RuntimeError(f'Invalid base model type: {base_model_type}')

    entities_dataset = ZeshelEntitiesDataset(data_dir,
                                             split='val',
                                             tokenizer=tokenizer,
                                             base_model_type=base_model_type)

    logger.info(f'Num entities: {len(entities_dataset)}')
    entities_loader = torch.utils.data.DataLoader(entities_dataset, batch_size=batch_size, num_workers=12)

    all_embeddings = []
    all_ids = []
    with torch.no_grad():
        for ids, entities_inputs in tqdm(entities_loader):
            entities_inputs = {k: v.to(DEVICE) for (k, v) in entities_inputs.items()}
            entity_embeddings = model.get_entity_embeddings(entities_inputs).cpu().numpy()
            all_embeddings.append(entity_embeddings)
            all_ids += ids

    all_embeddings = np.vstack(all_embeddings)
    print(all_embeddings.shape)
    np.save('zeshel_entity_embeddings_val', {'embeddings': all_embeddings, 'ids': all_ids})


def embedd_mentions(checkpoint_path: str, data_dir: str, batch_size: int, base_model_type: str, split: str):
    model = BiEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device('cpu'),
        base_model_type=base_model_type
    )
    model.eval()
    model.to(DEVICE)

    tokenizer = get_tokenizer(base_model_type)
    mentions_dataset = ZeshelDataset(data_dir, split=split, tokenizer=tokenizer, base_model_type=base_model_type)

    logger.info(f'Num mentions: {len(mentions_dataset)}')
    mentions_loader = torch.utils.data.DataLoader(mentions_dataset, batch_size=batch_size, num_workers=12)

    all_embeddings = []
    entity_ids = []
    mention_ids = []
    with torch.no_grad():
        for batch in tqdm(mentions_loader):
            entity_ids += batch['entity_document_ids']
            mention_ids += batch['mention_document_ids']
            mention_inputs = batch['mention_inputs']
            mention_inputs = {k: v.to(DEVICE) for (k, v) in mention_inputs.items()}
            embeddings = model.get_mention_embeddings(mention_inputs).cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    print(all_embeddings.shape)
    np.save(f'zeshel_mention_embeddings_{split}', {
        'embeddings': all_embeddings,
        'entity_ids': entity_ids,
        'mention_ids': mention_ids})


def compute_embeddings(args: Namespace, split: str):
    logger.info(f"Computing mention embeddings.")
    embedd_mentions(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        base_model_type=args.base_model_type,
        split=split)

    logger.info(f"Computing entity embeddings.")
    embedd_entities(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        base_model_type=args.base_model_type,
        split=split)


def main():
    input_args = parse_cli_args()
    compute_embeddings(input_args, 'val')
    compute_embeddings(input_args, 'test')


if __name__ == '__main__':
    main()
