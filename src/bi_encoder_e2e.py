from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import DistilBertModel, AdamW


class BiEncoderE2E(pl.LightningModule):
    def __init__(self):
        super(BiEncoderE2E, self).__init__()

        self.context_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Entity embedder
        self.entity_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.fc_me = nn.Linear(768, 128)
        self.fc_ee = nn.Linear(768, 128)

        # mention detection params
        self.fc_mention = torch.nn.Linear(768, 3)  # mention scores for start, end, and middle.

    def get_entity_embeddings(self, entity_inputs):
        entity_inputs = {k: v.to(self.device) for k, v in entity_inputs.items()}
        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        ee = self.fc_ee(ee)
        return ee

    def get_mention_embeddings(self, mention_inputs):
        mention_inputs: Dict[str, Tensor] = {k: v.to(self.device) for k, v in mention_inputs.items()}
        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        me = self.fc_me(me)
        return me

    def forward(self, inputs, **kwargs):
        context_mask = inputs['context_attention_mask']

        context_embs = self.context_embedder(
            input_ids=inputs['context_ids'],
            attention_mask=context_mask
        ).last_hidden_state  # (bs, seqlen, embsize)

        (bs, seqlen, embsize) = context_embs.shape

        # (bs, seqlen, 3)
        mention_bounds_logits = self.fc_mention(context_embs)
        assert mention_bounds_logits.shape == torch.Size([bs, seqlen, 3])

        # (bs, seqlen, 1); (bs, seqlen, 1); (bs, seqlen, 1)
        starts, ends, middles = mention_bounds_logits.split(1, dim=-1)

        # (bs, seqlen)
        starts = starts.squeeze(-1)
        ends = ends.squeeze(-1)
        middles = middles.squeeze(-1)

        # impossible to choose masked tokens as starts/ends of spans
        starts[~context_mask] = -float("Inf")
        ends[~context_mask] = -float("Inf")
        middles[~context_mask] = -float("Inf")

        #########################
        # Compute mention scores.
        # Mention score for span [i,j] is start_i + middle_i+1 + middle_i+2 + ... + middle_j-1 + end_j
        #########################

        # First sum start_i's and end_j's
        # (bs, seqlen, seqlen)
        mention_scores = starts.unsqueeze(2) + ends.unsqueeze(1)
        assert mention_scores.shape == torch.Size([bs, seqlen, seqlen])

        # (bs, seqlen, seqlen)
        middle_sums = torch.zeros(mention_scores.size(), dtype=mention_scores.dtype).to(mention_scores.device)

        # add ends
        middle_cumsum = torch.zeros(bs, dtype=mention_scores.dtype).to(mention_scores.device)

        for i in range(seqlen):
            middle_cumsum += middles[:, i]
            middle_sums[:, :, i] += middle_cumsum.unsqueeze(-1)

        # subtract starts
        middle_cumsum = torch.zeros(bs, dtype=mention_scores.dtype).to(mention_scores.device)

        for i in range(seqlen - 1):
            middle_cumsum += middles[:, i]
            middle_sums[:, (i + 1), :] -= middle_cumsum.unsqueeze(-1)

        # (bs, seqlen, seqlen)
        mention_scores += middle_sums

        # Get matrix of mention bounds
        # (seqlen, seqlen, 2) -- tuples of [start_idx, end_idx]
        mention_bounds = torch.stack([
            torch.arange(seqlen).unsqueeze(-1).expand(seqlen, seqlen),
            # start idxs
            torch.arange(seqlen).unsqueeze(0).expand(seqlen, seqlen),
            # end idxs
        ], dim=-1).to(mention_scores.device)

        # (seqlen, seqlen)
        mention_sizes = mention_bounds[:, :, 1] - mention_bounds[:, :, 0] + 1  # (+1 as ends are inclusive)

        # Remove invalids (startpos > endpos, endpos > seqlen) and renormalize

        # (bs, seqlen, seqlen)
        valid_mask = (mention_sizes.unsqueeze(0) > 0) & context_mask.unsqueeze(1)

        # (bs, seqlen, seqlen)
        mention_scores[~valid_mask] = -float("inf")  # invalids have logprob=-inf (p=0)

        # (bs, seqlen * seqlen)
        mention_scores = mention_scores.view(bs, -1)

        # (bs, seqlen * seqlen, 2)
        mention_bounds = mention_bounds.view(-1, 2)
        mention_bounds = mention_bounds.unsqueeze(0).expand(bs, seqlen, 2)

        return mention_scores, mention_bounds

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
