import torch
import math
from torch import nn
from .util import get_basic_c
from .encoding import PATCHES_COUNT

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
ATTR_CLASSES = {
    "id": 120,
    "lane": 5,
    "roles": 7,
    "specialities": 16,
    "patch": PATCHES_COUNT
}
def scaled_sqrt_factory(scale=1):
    return lambda x: int(scale*math.ceil(math.sqrt(x)))

def create_embedding_sizes(
    columns, 
    f=scaled_sqrt_factory(2)
):
    if isinstance(columns[0], int):
        classes = columns
        return [(cl, f(cl)) for cl in classes]
    else:
        embedded = []
        classes = []
        for c in columns:
            bc = get_basic_c(c)
            try:
                index = embedded.index(bc)
                classes.append(index)
            except ValueError as ex:
                cl = ATTR_CLASSES[get_basic_c(c)]
                classes.append((cl, f(cl)))
            embedded.append(bc)
        return classes


class HeroEmbedder(torch.nn.Module):
    def __init__(self, sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embeddings = []
        for s in sizes:
            if isinstance(s, int):
                embedding = embeddings[s]
            else:
                embedding = torch.nn.Embedding(*s)
            embeddings.append(embedding)
        embeddings = torch.nn.ModuleList(embeddings)
        self.embeddings = embeddings
        self.main_dim = embeddings[0].weight.shape[-1]
        self.dim = sum(e.weight.shape[-1] for e in embeddings)

    def embed_batch(self, encoded_tensor):
        split_encoded = torch.split(encoded_tensor, 1, dim=-1)
        split_encoded = [torch.squeeze(e, dim=-1) for e in split_encoded]
        #print(len(split_encoded), split_encoded[0].shape)
        split_embed = [
            self.embeddings[i](split_encoded[i]) 
            for i in range(len(split_encoded))
        ]
        embedded = torch.cat(split_embed, dim=-1)
        #print(split_embed.shape)
        return embedded

    def forward(self, encoded_tensor):
        return self.embed_batch(encoded_tensor)

    def __call__(self, encoded_tensor):
        return self.embed_batch(encoded_tensor)
    
    def reverse(self, sample):
        sample = sample[..., :self.main_dim]
        distance = torch.norm(self.embeddings[0].weight.data - sample, dim=1)
        nearest = torch.argmin(distance)
        return nearest
    