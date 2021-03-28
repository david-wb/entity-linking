from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import BertModel, AdamW, AutoModel, RobertaModel

from src.enums import BaseModelType


class BiEncoder(pl.LightningModule):
    def __init__(self, base_model_type: str):
        super(BiEncoder, self).__init__()

        self.base_model_type = base_model_type

        if base_model_type == BaseModelType.BERT_BASE.name:
            # Mention embedder
            self.mention_embedder = BertModel.from_pretrained('bert-base-uncased')
            # Entity embedder
            self.entity_embedder = BertModel.from_pretrained('bert-base-uncased')
        elif base_model_type == BaseModelType.ROBERTA_BASE.name:
            # Mention embedder
            self.mention_embedder = RobertaModel.from_pretrained('roberta-base')
            # Entity embedder
            self.entity_embedder = RobertaModel.from_pretrained('roberta-base')
        elif base_model_type == BaseModelType.DECLUTR_BASE.name:
            # Mention embedder
            self.mention_embedder = AutoModel.from_pretrained("johngiorgi/declutr-base")
            # Entity embedder
            self.entity_embedder = AutoModel.from_pretrained("johngiorgi/declutr-base")
        else:
            raise RuntimeError(f'Invalid base model type: {base_model_type}')

        self.fc_me = nn.Linear(768, 128)
        self.fc_ee = nn.Linear(768, 128)

    def get_entity_embeddings(self, entity_inputs):
        entity_inputs = {k: v.to(self.device) for k, v in entity_inputs.items()}

        if self.base_model_type == BaseModelType.BERT_BASE.name:
            ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
            ee = self.fc_ee(ee)
        elif self.base_model_type == BaseModelType.ROBERTA_BASE.name:
            sequence_output = self.entity_embedder(**entity_inputs)[0]
            ee = torch.sum(
                sequence_output * entity_inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(entity_inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
            ee = self.fc_ee(ee)
        elif self.base_model_type == BaseModelType.DECLUTR_BASE.name:
            sequence_output = self.entity_embedder(**entity_inputs)[0]
            ee = torch.sum(
                sequence_output * entity_inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(entity_inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
            ee = self.fc_ee(ee)
        else:
            raise RuntimeError(f'Invalid base model type: {self.base_model_type}')

        return ee

    def get_mention_embeddings(self, mention_inputs):
        mention_inputs: Dict[str, Tensor] = {k: v.to(self.device) for k, v in mention_inputs.items()}

        if self.base_model_type == BaseModelType.BERT_BASE.name:
            me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
            me = self.fc_me(me)
        elif self.base_model_type == BaseModelType.ROBERTA_BASE.name:
            sequence_output = self.mention_embedder(**mention_inputs)[0]
            me = torch.sum(
                sequence_output * mention_inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(mention_inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
            me = self.fc_me(me)
        elif self.base_model_type == BaseModelType.DECLUTR_BASE.name:
            sequence_output = self.mention_embedder(**mention_inputs)[0]
            me = torch.sum(
                sequence_output * mention_inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(mention_inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
            me = self.fc_me(me)
        else:
            raise RuntimeError(f'Invalid base model type: {self.base_model_type}')

        return me

    def forward(self, mention_inputs, entity_inputs=None, **kwargs):
        me = self.get_mention_embeddings(mention_inputs)

        if entity_inputs:
            ee = self.get_entity_embeddings(entity_inputs)
            return me, ee

        return me

    def training_step(self, batch, batch_idx):
        me, ee = self.forward(**batch)
        scores = me.mm(ee.t())

        bs = ee.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        me, ee = self.forward(**batch)
        scores = me.mm(ee.t())

        bs = ee.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer
