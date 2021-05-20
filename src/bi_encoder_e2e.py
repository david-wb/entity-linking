from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertModel, AdamW

from src.span_utils import batched_span_select
from src.utils import batch_reshape_mask_left


class BiEncoderE2E(pl.LightningModule, ABC):
    def __init__(self):
        super(BiEncoderE2E, self).__init__()

        self.context_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.mention_detector_head = MentionDetectorHead()
        self.mention_encoder_head = MentionSpanEncoderHead()
        self.entity_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def get_entity_embeddings(self, input_ids, attention_mask):
        bs = input_ids.shape[0]
        embs = self.entity_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        assert embs.shape == torch.Size([bs, 768])
        return embs

    def get_context_embeddings(self, input_ids, attention_mask):
        bs, seqlen = input_ids.shape
        embs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        assert embs.shape == torch.Size([bs, seqlen, 768])
        return embs

    def forward(self,
                context_ids,
                context_mask,
                entity_ids,
                entity_mask,
                all_entities_mask,
                mention_starts,
                mention_ends,
                mention_middles,
                mention_bounds,
                mention_bounds_mask):

        context_outs = self.forward_context(context_ids=context_ids,
                                            context_mask=context_mask,
                                            true_mention_bounds=mention_bounds,
                                            true_mention_bounds_mask=mention_bounds_mask)

        bs = entity_ids.shape[0]
        max_num_entities = entity_ids.shape[1]

        # (bs, max_num_entities, embed_dim)
        entity_embs = self.get_entity_embeddings(
            input_ids=entity_ids.view(-1, 512),
            attention_mask=entity_mask.view(-1, 512)
        )

        entity_embs = entity_embs.view(bs, max_num_entities, -1)

        return context_outs, entity_embs

    def forward_context(
            self,
            context_ids,
            context_mask,
            true_mention_bounds=None,
            true_mention_bounds_mask=None,
            max_mentions=50,
            topk_threshold=-4.5
    ):
        """
        If true_mention_bounds is set, returns mention embeddings of passed-in mention bounds
        Otherwise, uses top-scoring mentions
        """

        (bs, seqlen) = context_ids.shape

        # (bs, seqlen, embed_size)
        context_embs = self.get_context_embeddings(context_ids, context_mask)
        mention_scores, mention_bounds = self.mention_detector_head(context_embs, context_mask)

        top_mention_bounds = None
        top_mention_scores = None
        top_mention_mask = None
        extra_rets = {}
        if true_mention_bounds is None:
            (
                top_mention_scores, top_mention_bounds, top_mention_mask, all_mention_mask,
            ) = self.prune_predicted_mentions(mention_scores, mention_bounds, max_mentions, topk_threshold)
            extra_rets['mention_scores'] = top_mention_scores.view(-1)
            extra_rets['all_mention_mask'] = all_mention_mask

        if top_mention_bounds is None:
            # use true mention
            top_mention_bounds = true_mention_bounds
            top_mention_mask = true_mention_bounds_mask

        assert top_mention_bounds is not None
        assert top_mention_mask is not None

        # (bs, max_num_mentions, embed_size)
        mention_embs = self.mention_encoder_head(context_embs, top_mention_bounds)

        return {
            "mention_embs": mention_embs,
            "mention_scores": top_mention_scores,
            "mention_bounds": top_mention_bounds,
            "mention_masks": top_mention_mask,
            "mention_dims": torch.tensor(top_mention_mask.size()).unsqueeze(0).to(mention_embs.device),
        }

    def prune_predicted_mentions(self,
                                 mention_scores,
                                 mention_bounds,
                                 num_cand_mentions,
                                 threshold):
        '''
            Prunes mentions based on mention scores/logits (by either
            `threshold` or `num_cand_mentions`, whichever yields less candidates)
        Inputs:
            mention_scores: torch.FloatTensor (bsz, num_total_mentions)
            mention_bounds: torch.IntTensor (bsz, num_total_mentions)
            num_cand_mentions: int
            threshold: float
        Returns:
            torch.FloatTensor(bsz, max_num_pred_mentions): top mention scores/logits
            torch.IntTensor(bsz, max_num_pred_mentions, 2): top mention boundaries
            torch.BoolTensor(bsz, max_num_pred_mentions): mask on top mentions
            torch.BoolTensor(bsz, total_possible_mentions): mask for reshaping from total possible mentions -> max # pred mentions
        '''
        # (bsz, num_cand_mentions); (bsz, num_cand_mentions)
        top_mention_scores, mention_pos = mention_scores.topk(num_cand_mentions, sorted=True)
        # (bsz, num_cand_mentions, 2)
        #   [:,:,0]: index of batch
        #   [:,:,1]: index into top mention in mention_bounds
        mention_pos = torch.stack(
            [torch.arange(mention_pos.size(0)).to(mention_pos.device).unsqueeze(-1).expand_as(mention_pos),
             mention_pos], dim=-1)
        # (bsz, num_cand_mentions)
        top_mention_pos_mask = torch.sigmoid(top_mention_scores).log() > threshold
        # (total_possible_mentions, 2)
        #   tuples of [index of batch, index into mention_bounds] of what mentions to include
        mention_pos = mention_pos[top_mention_pos_mask | (
            # 2nd part of OR: if nothing is > threshold, use topK that are > -inf
                ((top_mention_pos_mask.sum(1) == 0).unsqueeze(-1)) & (top_mention_scores > -float("inf"))
        )]
        mention_pos = mention_pos.view(-1, 2)
        # (bsz, total_possible_mentions)
        #   mask of possible logits
        mention_pos_mask = torch.zeros(mention_scores.size(), dtype=torch.bool).to(mention_pos.device)
        mention_pos_mask[mention_pos[:, 0], mention_pos[:, 1]] = 1
        # (bsz, max_num_pred_mentions, 2)
        chosen_mention_bounds, chosen_mention_mask = batch_reshape_mask_left(mention_bounds, mention_pos_mask,
                                                                             pad_idx=0)
        # (bsz, max_num_pred_mentions)
        chosen_mention_scores, _ = batch_reshape_mask_left(mention_scores, mention_pos_mask, pad_idx=-float("inf"),
                                                           left_align_mask=chosen_mention_mask)
        return chosen_mention_scores, chosen_mention_bounds, chosen_mention_mask, mention_pos_mask

    def get_span_loss(
        self, true_mention_bounds, true_mention_bounds_mask, mention_scores, mention_bounds,
    ):
        """
        true_mention_bounds (bs, num_mentions, 2)
        true_mention_bounds_mask (bs, num_mentions):
        mention_scores (bs, num_spans)
        mention_bounds (bs, num_spans, 2)
        """
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")

        true_mention_bounds[~true_mention_bounds_mask] = -1  # ensure don't select masked to score
        # triples of [ex in batch, mention_idx in true_mention_bounds, idx in mention_bounds]
        # use 1st, 2nd to index into true_mention_bounds, 1st, 3rd to index into mention_bounds
        true_mention_pos_idx = ((
                                        mention_bounds.unsqueeze(1) - true_mention_bounds.unsqueeze(2)
                                # (bs, num_mentions, start_pos * end_pos, 2)
                                ).abs().sum(-1) == 0).nonzero()
        # true_mention_pos_idx should have 1 entry per masked element
        # (num_true_mentions [~true_mention_bounds_mask])
        true_mention_pos = true_mention_pos_idx[:, 2]

        # (bs, total_possible_spans)
        true_mention_binary = torch.zeros(mention_scores.size(), dtype=mention_scores.dtype).to(
            true_mention_bounds.device)
        true_mention_binary[true_mention_pos_idx[:, 0], true_mention_pos_idx[:, 2]] = 1

        # prune masked spans
        mask = mention_scores != -float("inf")
        masked_mention_scores = mention_scores[mask]
        masked_true_mention_binary = true_mention_binary[mask]

        # (bs, total_possible_spans)
        span_loss = loss_fct(masked_mention_scores, masked_true_mention_binary)

        return span_loss

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


class MentionDetectorHead(pl.LightningModule):
    def __init__(self):
        super(MentionDetectorHead, self).__init__()

        # mention scores for start, end, and middle.
        self.fc_mention = torch.nn.Linear(768, 3)

    def forward(self, bert_output: torch.Tensor, context_mask):
        (bs, seqlen, embed_dim) = bert_output.shape

        # (bs, seqlen, 3)
        mention_bounds_logits = self.fc_mention(bert_output)
        assert mention_bounds_logits.shape == torch.Size([bs, seqlen, 3])

        # (bs, seqlen, 1), (bs, seqlen, 1), (bs, seqlen, 1)
        starts, ends, middles = mention_bounds_logits.split(1, dim=-1)

        # (bs, seqlen)
        starts = starts.squeeze(-1)
        ends = ends.squeeze(-1)
        middles = middles.squeeze(-1)

        # Mask out invalid mention positions. (Causes them to have zero probability)
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
        valid_mask = (mention_sizes.unsqueeze(0) > 0) & (context_mask.unsqueeze(1) > 0)

        # (bs, seqlen, seqlen)
        mention_scores[~valid_mask] = -float("inf")  # invalids have logprob=-inf (p=0)

        # (bs, seqlen * seqlen)
        mention_scores = mention_scores.view(bs, -1)

        # (seqlen * seqlen, 2)
        mention_bounds = mention_bounds.view(-1, 2)
        # (bs, seqlen * seqlen, 2)
        mention_bounds = mention_bounds.unsqueeze(0).expand(bs, seqlen * seqlen, 2)

        return mention_scores, mention_bounds


class MentionSpanEncoderHead(nn.Module, ABC):
    """
    Computes the average embedding for each span of tokens.
    """
    def __init__(self):
        super(MentionSpanEncoderHead, self).__init__()

    def forward(self, bert_output: torch.Tensor, mention_bounds: torch.LongTensor):
        """
        bert_output
            (bs, seqlen, embed_dim)
        mention_bounds: both bounds are inclusive [start, end]
            (bs, num_spans, 2)
        """
        (
            embedding_ctxt,  # (bs, num_spans, max_batch_span_width, embed_dim)
            mask,  # (bs, num_spans, max_batch_span_width)
        ) = batched_span_select(
            bert_output,  # (bs, sequence_length, embed_dim)
            mention_bounds,  # (bs, num_spans, 2)
        )
        embedding_ctxt[~mask] = 0  # 0 out masked elements
        # embedding_ctxt = (bs, num_spans, max_batch_span_width, embed_dim)
        embedding_ctxt = embedding_ctxt.sum(2) / mask.sum(2).float().unsqueeze(-1)
        # embedding_ctxt = (bs, num_spans, embed_dim)

        return embedding_ctxt
