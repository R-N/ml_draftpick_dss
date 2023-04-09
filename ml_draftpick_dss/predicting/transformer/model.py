import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchinfo import summary
import math
from ..modules import GlobalPooling1D, Residual, create_mlp_stack, RepeatExpander, MLPExpander
from ..embedding import PositionalEncoding, HeroEmbedder

class ResultPredictorModel(nn.Module):
    def __init__(
        self, 
        embedding,
        encoder_kwargs,
        tf_encoder_kwargs,
        tf_decoder_kwargs,
        final_kwargs,
        head_kwargs,
        reducer_kwargs={},
        pooling=GlobalPooling1D(),
        pos_encoder=False,
        dropout=0.1,
        bidirectional=False,
        dim=3,
        expander_kwargs={}
    ):
        super().__init__()
        self.name = "predictor_tf"
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
        self._create_encoder(**encoder_kwargs)
        if self.dim == 2:
            self.d_tf = tf_encoder_kwargs["n_heads"]
        else:
            self.d_tf = self.encoder.dim
        self.pos_encoder = PositionalEncoding(self.d_tf, dropout) if pos_encoder else None
        self._create_tf_encoder(**tf_encoder_kwargs)
        self._create_tf_decoder(**tf_decoder_kwargs)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)
        if self.dim == 2:
            self.d_reducer = self.d_tf
            self.d_final = self._calc_d_final(self.encoder.dim)
            if expander_kwargs:
                self.expander = MLPExpander(self.tf_encoder_heads, d_input=self.encoder.dim, d_output=self.encoder.dim, **expander_kwargs)
            else:
                self.expander = RepeatExpander(self.tf_encoder_heads)
            self._create_reducer(**reducer_kwargs)
        else:
            self.d_final = self._calc_d_final(self.d_tf)
        self._create_final(**final_kwargs)
        self._create_heads(**head_kwargs)

    def _create_encoder(self, d_hid, d_output=0, **kwargs):
        d_output = d_output or d_hid
        self.encoder = create_mlp_stack(self.d_embed, d_hid, d_output, **kwargs)

    def _create_tf_encoder(self, n_heads, d_hid, n_layers, dropout=0.1, activation=torch.nn.ReLU):
        #d_model = n_heads*self.d_tf
        d_model = self.d_tf
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout=dropout, activation=activation(), batch_first=True)
        self.tf_encoder_heads = n_heads
        self.tf_encoder = TransformerEncoder(encoder_layers, n_layers)

    def _create_tf_decoder(self, n_heads, d_hid, n_layers, dropout=0.1, activation=torch.nn.ReLU):
        #d_model = n_heads*self.d_tf
        d_model = self.d_tf
        decoder_layers = TransformerDecoderLayer(d_model, n_heads, d_hid, dropout=dropout, activation=activation(), batch_first=True)
        self.tf_decoder_heads = n_heads
        self.tf_decoder = TransformerDecoder(decoder_layers, n_layers)

    def _create_reducer(self, d_hid, n_layers=1, activation=torch.nn.Identity, bias=False, dropout=0):
        d_out = 1 if self.dim == 2 else self.d_final
        self.reducer = create_mlp_stack(self.d_reducer, d_hid, d_out, n_layers, activation=activation, bias=bias, dropout=dropout)
    
    def _calc_d_final(self, d_final):
        d_final = (2 if self.bidirectional else 1) * d_final
        d_final = (1 if (
            self.pooling and not isinstance(self.pooling, torch.nn.Flatten)
        ) else 5) * d_final
        return d_final

    def _create_final(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        self.final = create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers, activation=activation, bias=bias, dropout=dropout)

    def _create_heads(self, d_hid=0, n_layers=1, heads=["victory", "score", "duration"], activation=torch.nn.ReLU, bias=True, dropout=0.1):
        self.head_labels = heads
        d_hid = d_hid or self.d_final
        self.heads = [
            nn.Sequential(*[
                create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
                nn.Sequential(*[
                    nn.Dropout(dropout),
                    nn.Linear(self.d_final, 1, bias=bias),
                    nn.Tanh()
                ])
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
        memory = self.tf_encoder(src)#, src_mask)
        tgt = self.tf_decoder(tgt, memory)
        return tgt
    
    def pos_encode(self, x):
        if self.pos_encoder:
            x = x * math.sqrt(self.d_embed)
            x = self.pos_encoder(x)
        return x

    def forward(self, left, right):

        left = self.embedding(left)
        right = self.embedding(right)

        left = self.encoder(left)
        right = self.encoder(right)

        if self.dim == 2:
            left = self.expander(left)
            right = self.expander(right)

        left = self.pos_encode(left)
        right = self.pos_encode(right)
        
        if self.bidirectional:
            left = self.transform(left, right)
            right = self.transform(right, left)
            tgt = torch.cat([left, right], dim=-1)
        else:
            tgt = self.transform(left, right)

        if self.dim == 2:
            tgt = self.reducer(tgt)

        if self.dim == 2:
            tgt = torch.squeeze(tgt, -1)
        else:
            tgt = self.pooling(tgt)

        tgt = self.final(tgt)

        output = [f(tgt) for f in self.heads]
        return output
    
    def summary(self, batch_size=32, team_size=5, dim=6, dtype=torch.int):
        return summary(
            self, 
            [(batch_size, team_size, dim) for i in range(2)], 
            dtypes=[dtype, dtype]
        )

