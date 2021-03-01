import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, AdamW


class BiEncoder(pl.LightningModule):
    def __init__(self):
        super(BiEncoder, self).__init__()

        # Mention embedder
        self.mention_embedder = BertModel.from_pretrained('bert-base-uncased')
        self.fc_me = nn.Linear(768, 128)

        # Entity embedder
        self.entity_embedder = BertModel.from_pretrained('bert-base-uncased')
        self.fc_ee = nn.Linear(768, 128)

    def get_entity_embeddings(self, entity_inputs):
        entity_inputs = {k: v.to(self.device) for k, v in entity_inputs.items()}
        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        ee = self.fc_ee(ee)

        return ee

    def get_mention_embeddings(self, mention_inputs):
        mention_inputs = {k: v.to(self.device) for k, v in mention_inputs.items()}
        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        me = self.fc_me(me)

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
