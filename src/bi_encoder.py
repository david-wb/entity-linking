import torch
from torch import nn
from transformers import DistilBertModel
import torch.nn.functional as F


class BiEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super(BiEncoder, self).__init__()

        self.device = device

        # Mention embedder
        self.mention_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.mention_embedder.train()
        self.fc_me = nn.Linear(768, 64)

        # Entity embedder
        self.entity_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.entity_embedder.train()
        self.fc_ee = nn.Linear(768, 64)
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    # x represents our data
    def forward(self, mention_inputs, entity_inputs, negative_entity_inputs):
        mention_inputs = {k: v.to(self.device) for k,v in mention_inputs.items()}
        entity_inputs = {k: v.to(self.device) for k,v in entity_inputs.items()}
        negative_entity_inputs = {k: v.to(self.device) for k,v in negative_entity_inputs.items()}

        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        me = self.fc_me(me)
        me_norm = me.norm(p=2, dim=1, keepdim=True)
        me = me.div(me_norm)

        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        ee = self.fc_ee(ee)
        ee_norm = ee.norm(p=2, dim=1, keepdim=True)
        ee = ee.div(ee_norm)

        nee = self.entity_embedder(**negative_entity_inputs).last_hidden_state[:, 0]
        nee = self.fc_ee(nee)
        nee_norm = nee.norm(p=2, dim=1, keepdim=True)
        nee = nee.div(nee_norm)

        scores = me.mm(ee.t())

        bs = ee.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")

        #neg_scores = torch.sum(me * nee, dim=-1)
        #loss = torch.mean(-scores + neg_scores)
        return me, ee, loss
