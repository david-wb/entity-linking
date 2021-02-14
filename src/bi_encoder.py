import torch
from torch import nn
from transformers import DistilBertModel


class BiEncoder(nn.Module):
    def __init__(self):
        super(BiEncoder, self).__init__()

        # Mention embedder
        self.mention_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.mention_embedder.train()

        # Entity embedder
        self.entity_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.entity_embedder.train()

    # x represents our data
    def forward(self, mention_inputs, entity_inputs, negative_entity_inputs):
        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        nee = self.entity_embedder(**negative_entity_inputs).last_hidden_state[:, 0]

        scores = torch.matmul(me, ee.T)
        neg_scores = torch.matmul(me, nee.T)
        loss = torch.mean(-scores + neg_scores)
        return me, ee, loss
