import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchinfo import summary
import math
from ..modules import GlobalPooling1D, Residual, create_mlp_stack
from ..embedding import PositionalEncoding, HeroEmbedder


class ResultPredictorModel(nn.Module):
    def __init__(
        self, 
        embedding,
        encoder_kwargs,
        decoder_kwargs,
        final_kwargs,
        head_kwargs,
        pooling=GlobalPooling1D(),
        pos_encoder=False,
        dropout=0.1,
        bidirectional=False,
        dim=3
    ):
        super().__init__()
        if isinstance(embedding, torch.nn.Module):
            self.embedding = embedding
            self.d_embed = self.embedding.dim
        elif isinstance(embedding, int):
            self.d_embed = embedding
            self.embedding = nn.Identity()
        elif embedding and hasattr(embedding, "__iter__"):
            self.embedding = HeroEmbedder(embedding)
            self.d_embed = self.embedding.dim
        self.dim = dim
        self.model_type = 'Transformer'
        self.bidirectional = bidirectional
        self.pos_encoder = PositionalEncoding(self.d_embed, dropout) if pos_encoder else None
        self._create_encoder(**encoder_kwargs)
        self._create_decoder(**decoder_kwargs)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self._create_final(**final_kwargs)
        self._create_heads(**head_kwargs)

    def _create_encoder(self, n_heads, d_hid, n_layers, dropout=0.1):
        d = self.d_embed if self.dim == 3 else 1
        encoder_layers = TransformerEncoderLayer(self.d_embed, n_heads, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, n_layers)

    def _create_decoder(self, n_heads, d_hid, n_layers, dropout=0.1):
        d = self.d_embed if self.dim == 3 else 1
        decoder_layers = TransformerDecoderLayer(self.d_embed, n_heads, d_hid, dropout, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, n_layers)

    def _create_final(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        d_final = self.d_embed
        d_final = (2 if self.bidirectional else 1) * d_final
        d_final = (1 if (
            self.pooling and not isinstance(self.pooling, torch.nn.Flatten)
        ) else 5) * d_final

        self.final = create_mlp_stack(d_final, d_hid, d_final, n_layers, activation=activation, bias=bias, dropout=dropout)
        self.d_final = d_final
        return d_final

    def _create_heads(self, heads=["victory", "score", "duration"], activation=nn.Tanh, bias=True, dropout=0.1):
        self.head_labels = heads
        self.heads = [
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.d_final, 1, bias=bias),
                activation()
            ]) for i in range(len(heads))
        ]
    
    def init_weights(self, layers=None, initrange=0.1):
        layers = layers or [
            self.final,
            self.heads,
        ]
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        for layer in layers:
            if hasattr(layer, "__iter__"):
                self.init_weights(layer)
            else:
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data.zero_()
                if hasattr(layer, "weight") and layer.weight is not None:
                    layer.weight.data.uniform_(-initrange, initrange)
    
    def transform(self, src, tgt):
        memory = self.encoder(src)#, src_mask)
        tgt = self.decoder(tgt, memory)
        return tgt
    
    def pos_encode(self, x):
        if self.pos_encoder:
            x = x * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
        return x

    def forward(self, left, right):

        left = self.embedding(left)
        right = self.embedding(right)

        if self.dim == 2:
            left = left[:, :, None]
            right = right[:, :, None]

        left = self.pos_encode(left)
        right = self.pos_encode(right)
        
        if self.bidirectional:
            left = self.transform(left, right)
            right = self.transform(right, left)
            tgt = torch.cat([left, right], dim=-1)
        else:
            tgt = self.transform(left, right)

        if self.dim == 2:
            left = torch.squeeze(left, start_dim=-1)
            right = torch.squeeze(right, start_dim=-1)
        else:
            tgt = self.pooling(tgt)
        tgt = self.final(tgt)
        output = [f(tgt) for f in self.heads]
        return output
    
    def summary(self, batch_size=32, team_size=5, dim=6):
        return summary(
            self, 
            [(batch_size, team_size, dim) for i in range(2)], 
            dtypes=[torch.int, torch.int]
        )

