import torch
from torch import nn
from transformers import BertModel, AdamW
import torch.nn.functional as F
import pytorch_lightning as pl


class BiEncoder(pl.LightningModule):
    def __init__(self, device='cpu'):
        super(BiEncoder, self).__init__()

        # Mention embedder
        self.mention_embedder = BertModel.from_pretrained('bert-base-uncased')
        self.fc_me = nn.Linear(768, 128)

        # Entity embedder
        self.entity_embedder = BertModel.from_pretrained('bert-base-uncased')
        self.fc_ee = nn.Linear(768, 128)

    # x represents our data
    def forward(self, mention_inputs, entity_inputs=None, negative_entity_inputs=None):
        mention_inputs = {k: v.to(self.device) for k,v in mention_inputs.items()}

        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        me = self.fc_me(me)
        me_norm = me.norm(p=2, dim=1, keepdim=True)
        me = me.div(me_norm)

        if entity_inputs:
            entity_inputs = {k: v.to(self.device) for k, v in entity_inputs.items()}

            ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
            ee = self.fc_ee(ee)
            ee_norm = ee.norm(p=2, dim=1, keepdim=True)
            ee = ee.div(ee_norm)

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
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer
