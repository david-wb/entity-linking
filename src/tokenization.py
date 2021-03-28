from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer

from src.enums import BaseModelType


def get_tokenizer(base_model_type: str):
    if base_model_type == BaseModelType.BERT_BASE.name:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif base_model_type == BaseModelType.ROBERTA_BASE.name:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif base_model_type == BaseModelType.DECLUTR_BASE.name:
        tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
    else:
        raise RuntimeError(f'Invalid base model type: {base_model_type}')
    return tokenizer
