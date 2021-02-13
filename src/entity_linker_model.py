from torch import nn
from transformers import DistilBertModel


class EntityLinker(nn.Module):
    def __init__(self):
        super(EntityLinker, self).__init__()

        # Mention embedder
        self.mention_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Entity embedder
        self.entity_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # x represents our data
    def forward(self, mention_inputs, entity_inputs, negative_entity_inputs):
        me = self.mention_embedder(**mention_inputs).last_hidden_state[:, 0]
        ee = self.entity_embedder(**entity_inputs).last_hidden_state[:, 0]
        nee = self.entity_embedder(**negative_entity_inputs).last_hidden_state[:, 0]
        return me, ee, nee
