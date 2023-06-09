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
        bidirectional=None,
        dim=3,
        expander_kwargs={}
    ):
        super().__init__()
        self.name = "predictor_tf"
        if isinstance(embedding, torch.nn.Module):
            self.embedding = embedding
            self.d_embed = self.embedding.dim
            self.d_input = self.embedding.input_dim
        elif isinstance(embedding, int):
            self.d_embed = embedding
            self.embedding = nn.Identity()
            self.d_input = self.d_embed
        elif embedding and hasattr(embedding, "__iter__"):
            self.embedding = HeroEmbedder(embedding)
            self.d_embed = self.embedding.dim
            self.d_input = self.embedding.input_dim
        self.dim = dim
        self.model_type = 'Transformer'
        """
        if isinstance(bidirectional, str) and bidirectional.lower() == "none":
            bidirectional = None
        """
        self.bidirectional = bidirectional
        self._create_encoder(**encoder_kwargs)
        if self.dim == 2:
            self.d_tf = tf_encoder_kwargs["n_heads"]
        else:
            self.d_tf = self.encoder.dim
        assert self.d_tf % 2 == 0
        self.pos_encoder = PositionalEncoding(self.d_tf, dropout) if pos_encoder else None
        self._create_tf_encoder(**tf_encoder_kwargs)
        self._create_tf_decoder(**tf_decoder_kwargs)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)
        if self.dim == 2:
            #self.d_reducer = self._calc_d_final(self.d_tf)
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
        d_final = (2 if self.bidirectional == "concat" else 1) * d_final
        d_final = (1 if (
            self.pooling and not isinstance(self.pooling, torch.nn.Flatten)
        ) else 5) * d_final
        return d_final

    def _create_final(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        self.final = create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers, activation=activation, bias=bias, dropout=dropout)

    def _create_heads(self, d_hid=0, n_layers=1, n_heads=3, activation=torch.nn.ReLU, activation_final=nn.Tanh, bias=True, dropout=0.1):
        d_hid = d_hid or self.d_final
        """
        self.head = nn.Sequential(*[
            create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.d_final, n_heads, bias=bias),
                nn.Tanh()
            ])
        ])
        """
        self.heads = [
            nn.Sequential(*[
                create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
                nn.Sequential(*[
                    nn.Dropout(dropout),
                    nn.Linear(self.d_final, 1, bias=bias),
                    activation_final()
                ])
            ]) for i in range(n_heads)
        ]
    
    def init_weights(self, layers=None, initrange=0.1):
        layers = layers or [
            self.final,
            self.head,
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
        try:
            tgt = self.tf_decoder(tgt, memory)
        except RuntimeError as ex:
            print(src.shape, tgt.shape, memory.shape)
            raise
        if self.dim == 2:
            tgt = self.reducer(tgt)
        return tgt
    
    def pos_encode(self, x):
        if self.pos_encoder:
            x = x * math.sqrt(self.d_embed)
            x = self.pos_encoder(x)
        return x

    def forward(self, left, right):

        left, right = (self.embedding(left), self.embedding(right))

        left, right = (self.encoder(left), self.encoder(right))

        if self.dim == 2:
            left, right = (self.expander(left), self.expander(right))

        left, right = (self.pos_encode(left), self.pos_encode(right))
        
        if "none" not in self.bidirectional:
            left, right = (self.transform(left, right), self.transform(right, left))
            if self.dim == 2:
                left, right = (torch.squeeze(left, -1), torch.squeeze(right, -1))
            else:
                left, right = (self.pooling(left), self.pooling(right))

            if self.bidirectional == "concat":
                tgt = torch.cat([left, right], dim=-1)
            elif self.bidirectional == "diff_left":
                tgt = left - right 
            elif self.bidirectional == "diff_right":
                tgt = right - left
            else:
                tgt = torch.stack([left, right])
                if self.bidirectional in ("avg", "average", "mean"):
                    tgt = torch.mean(tgt, dim=0)
                elif self.bidirectional == "prod":
                    tgt = torch.prod(tgt, dim=0)
                elif self.bidirectional == "max":
                    tgt = torch.max(tgt, dim=0)
                else:
                    raise ValueError(f"Unknown bidirectional option {self.bidirectional}")
            if isinstance(tgt, tuple):
                tgt = tgt[0]
        else:
            if "right" in self.bidirectional:
                left, right = right, left
            tgt = self.transform(left, right)
            if self.dim == 2:
                tgt = torch.squeeze(tgt, -1)
            else:
                tgt = self.pooling(tgt)

        try:
            tgt = self.final(tgt)
        except RuntimeError as ex:
            print(tgt.shape, self.d_final)
            raise

        #output = self.head(tgt)
        output = [f(tgt) for f in self.heads]
        return output
    
    def summary(self, batch_size=64, team_size=5, dtype=torch.int):
        return summary(
            self, 
            [(batch_size, team_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )

