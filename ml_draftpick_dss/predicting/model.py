import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchinfo import summary
import math
from .modules import PositionalEncoding, GlobalPooling1D
from .preparation import HeroEmbedder


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
    ):
        super().__init__()
        if isinstance(embedding, HeroEmbedder):
            self.embedding = embedding
            self.d_embed = self.embedding.dim
        elif isinstance(embedding, int):
            self.d_embed = embedding
            self.embedding = nn.Identity()
        elif embedding and hasattr(embedding, "__iter__"):
            self.embedding = HeroEmbedder(embedding)
            self.d_embed = self.embedding.dim
        self.model_type = 'Transformer'
        self.bidirectional = bidirectional
        self.pos_encoder = PositionalEncoding(self.d_embed, dropout) if pos_encoder else None
        self._create_encoder(**encoder_kwargs)
        self._create_decoder(**decoder_kwargs)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self._create_final(**final_kwargs)
        self._create_heads(**head_kwargs)

    def _create_encoder(self, n_head, d_hid, n_layers, dropout=0.1):
        encoder_layers = TransformerEncoderLayer(self.d_embed, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

    def _create_decoder(self, n_head, d_hid, n_layers, dropout=0.1):
        decoder_layers = TransformerDecoderLayer(self.d_embed, n_head, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers)

    def _create_final(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        d_final = self.d_embed
        d_final = (2 if self.bidirectional else 1) * d_final
        d_final = (1 if (
            self.pooling and not isinstance(self.pooling, torch.nn.Flatten)
        ) else 5) * d_final

        if n_layers == 0:
            self.final = nn.Identity()
        elif n_layers == 1:
            self.final = nn.Sequential(
                *[
                    nn.Dropout(dropout),
                    nn.Linear(d_final, d_final, bias=bias),
                    activation()
                ]
            )
        else:
            self.final = nn.Sequential(
                *[
                    nn.Dropout(dropout),
                    nn.Linear(d_final, d_hid, bias=bias),
                    activation(),
                    #nn.Dropout(dropout)
                ],
                *[
                    nn.Sequential(*[
                        nn.Dropout(dropout),
                        nn.Linear(d_hid, d_hid, bias=bias),
                        activation(),
                    ])
                    for i in range(max(0, n_layers-2))
                ],
                *[
                    nn.Dropout(dropout),
                    nn.Linear(d_hid, d_final, bias=bias),
                    activation(),
                    #nn.Dropout(dropout)
                ],
            )
        self.d_final = d_final
        return d_final

    def _create_heads(self, heads=["victory", "score", "duration"], activation=nn.Tanh, bias=True, dropout=0.1):
        self.head_labels = heads
        self.heads = [
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.final_dim, 1, bias=bias),
                activation()
            ]) for i in range(len(heads))
        ]
    
    def init_weights(self, layers=None, initrange=0.1):
        layers = layers or [
            self.decoder,
            self.victory_decoder,
            self.score_decoder,
            self.duration_decoder
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
        left = self.pos_encode(left)
        right = self.embedding(right)
        right = self.pos_encode(right)
        
        if self.bidirectional:
            left = self.transform(left, right)
            right = self.transform(right, left)
            tgt = torch.cat([left, right], dim=-1)
        else:
            tgt = self.transform(left, right)

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

