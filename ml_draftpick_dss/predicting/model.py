import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchinfo import summary
import math
from .modules import PositionalEncoding, GlobalPooling1D



class ResultPredictorModel(nn.Module):

    def __init__(self, 
        d_model, 
        d_hid=128,
        nlayers=2,
        nhead=2,
        d_final=2,
        embedder=None,
        dropout=0.1,
        pooling=GlobalPooling1D(),
        act_final=nn.ReLU,
        bidirectional=False,
        pos_encoder=False,
        bias_final=True
    ):
        super().__init__()
        if embedder:
            d_model = embedder.dim
        else:
            embedder = nn.Identity()
        self.model_type = 'Transformer'
        self.bidirectional = bidirectional
        self.pos_encoder = PositionalEncoding(d_model, dropout) if pos_encoder else None
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.d_model = d_model
        self.encoder = embedder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)

        final_dim = d_model
        final_dim = (2 if bidirectional else 1) * final_dim
        final_dim = (1 if pooling else 5) * final_dim
        if d_final == 0:
            self.decoder = nn.Identity()
        elif d_final == 1:
            self.decoder = nn.Sequential(
                *[
                    nn.Linear(final_dim, final_dim, bias=bias_final),
                    act_final()
                ]
            )
        else:
            self.decoder = nn.Sequential(
                *[
                    nn.Linear(final_dim, d_hid, bias=bias_final),
                    act_final(),
                    #nn.Dropout(dropout)
                ],
                *[
                    nn.Sequential(*[
                        nn.Linear(d_hid, d_hid, bias=bias_final),
                        act_final(),
                        nn.Dropout(dropout)
                    ])
                    for i in range(max(0, d_final-2))
                ],
                *[
                    nn.Linear(d_hid, final_dim, bias=bias_final),
                    act_final(),
                    #nn.Dropout(dropout)
                ],
            )
        self.victory_decoder = nn.Sequential(*[
            nn.Linear(final_dim, 1, bias=False),
            nn.Tanh()
        ])
        self.score_decoder = nn.Sequential(*[
            nn.Linear(final_dim, 1, bias=False),
            nn.Tanh()
        ])
        self.duration_decoder = nn.Sequential(*[
            nn.Linear(final_dim, 1, bias=False),
            nn.Tanh()
        ])

        #self.init_weights()
    
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
        memory = self.transformer_encoder(src)#, src_mask)
        tgt = self.transformer_decoder(tgt, memory)
        return tgt
    
    def pos_encode(self, x):
        if self.pos_encoder:
            x = x * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
        return x

    def forward(self, left, right):
        left = self.encoder(left)
        left = self.pos_encode(left)
        right = self.encoder(right)
        right = self.pos_encode(right)
        
        if self.bidirectional:
            left = self.transform(left, right)
            right = self.transform(right, left)
            tgt = torch.cat([left, right], dim=-1)
        else:
            tgt = self.transform(left, right)

        tgt = self.pooling(tgt)
        tgt = self.decoder(tgt)
        victory = self.victory_decoder(tgt)
        score = self.score_decoder(tgt)
        duration = self.duration_decoder(tgt)
        output = victory, score, duration
        #output = torch.cat(output, dim=-1)
        return output
    
    def summary(self, batch_size=32, team_size=5, dim=6):
        return summary(
            self, 
            [(batch_size, team_size, dim) for i in range(2)], 
            dtypes=[torch.int, torch.int]
        )

