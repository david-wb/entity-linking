import torch
from torch import nn
from transformers import DistilBertModel


class BiEncoder(nn.Module):
    def __init__(self):
        super(BiEncoder, self).__init__()

        # Mention embedder
        self.mention_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.mention_embedder.train()
        self.fc_me = nn.Linear(768, 64)

        # Entity embedder
        self.entity_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.entity_embedder.train()
        self.fc_ee = nn.Linear(768, 64)

    # x represents our data
    def forward(self, mention_inputs, entity_inputs, negative_entity_inputs):
        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        me = self.fc_me(me)

        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        ee = self.fc_ee(ee)

        nee = self.entity_embedder(**negative_entity_inputs).last_hidden_state[:, 0]
        nee = self.fc_ee(nee)

        scores = torch.sum(me * ee, dim=-1)
        neg_scores = torch.sum(me * nee, dim=-1)
        loss = torch.mean(-scores + neg_scores)
        return me, ee, loss