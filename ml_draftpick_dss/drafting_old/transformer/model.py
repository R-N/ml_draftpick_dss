import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchinfo import summary
import math
from ...predicting.modules import GlobalPooling1D, create_mlp_stack
from ...predicting.embedding import PositionalEncoding, HeroEmbedder

class TransformerModel(nn.Module):
    def __init__(
        self, 
        embedding,
        encoder_kwargs,
        tf_encoder_kwargs,
        tf_decoder_kwargs,
        pooling=GlobalPooling1D(),
        pos_encoder=True,
        dropout=0.1,
        bidirectional=None,
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
        self.model_type = 'Transformer'
        self.bidirectional = bidirectional
        self.encoder = self._create_encoder(**encoder_kwargs)
        self.d_tf = self.encoder.dim
        assert self.d_tf % 2 == 0
        self.pos_encoder = PositionalEncoding(self.d_tf, dropout) if pos_encoder else None
        self.tf_encoder = self._create_tf_encoder(**tf_encoder_kwargs)
        self.tf_decoder = self._create_tf_decoder(**tf_decoder_kwargs)
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)

    def _create_encoder(self, d_hid, d_output=0, **kwargs):
        d_output = d_output or d_hid
        return create_mlp_stack(self.d_embed, d_hid, d_output, **kwargs)

    def _create_tf_encoder(self, n_heads, d_hid, n_layers, dropout=0.1, activation=torch.nn.ReLU):
        #d_model = n_heads*self.d_tf
        d_model = self.d_tf
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout=dropout, activation=activation(), batch_first=True)
        return TransformerEncoder(encoder_layers, n_layers)

    def _create_tf_decoder(self, n_heads, d_hid, n_layers, dropout=0.1, activation=torch.nn.ReLU):
        #d_model = n_heads*self.d_tf
        d_model = self.d_tf
        decoder_layers = TransformerDecoderLayer(d_model, n_heads, d_hid, dropout=dropout, activation=activation(), batch_first=True)
        return TransformerDecoder(decoder_layers, n_layers)
    
    def pos_encode(self, x):
        if self.pos_encoder:
            x = x * math.sqrt(self.d_embed)
            x = self.pos_encoder(x)
        return x
    
    def transform(self, src, tgt):
        memory = self.tf_encoder(src)
        try:
            tgt = self.tf_decoder(tgt, memory)
        except RuntimeError as ex:
            print(src.shape, tgt.shape, memory.shape)
            raise
        return tgt
    
    
    def forward(self, left, right):
        left, right = (self.embedding(left), self.embedding(right))
        left, right = (self.encoder(left), self.encoder(right))
        left, right = (self.pos_encode(left), self.pos_encode(right))
        
        if "none" not in self.bidirectional:
            left, right = (
                self.transform(left, right), 
                self.transform(right, left)
            )
            left, right = (self.pooling(left), self.pooling(right))

            if self.bidirectional == "concat":
                tgt = torch.cat([left, right], dim=-1)
            elif self.bidirectional == "diff_left":
                tgt = left - right 
            elif self.bidirectional == "diff_right":
                tgt = right - left
            else:
                tgt = torch.stack([left, right])
                if self.bidirectional == "mean":
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
            tgt = self.pooling(tgt)

        return tgt

class DraftingAgentModel(nn.Module):
    def __init__(
        self, 
        embedding,
        tf_kwargs,
        final_kwargs,
        final_2_kwargs,
        head_kwargs,
        tf_ban_kwargs={},
        pooling=GlobalPooling1D(),
        pos_encoder=True,
        dropout=0.1,
        bidirectional=None,
        final_2_mode="double",
        v_pooling="mean",
        game=None
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
            self.embedding.dim = self.d_embed
            self.embedding.input_dim = self.d_input
        elif embedding and hasattr(embedding, "__iter__"):
            self.embedding = HeroEmbedder(embedding)
            self.d_embed = self.embedding.dim
            self.d_input = self.embedding.input_dim
        self.model_type = 'Transformer'
        self.final_2_mode = final_2_mode
        self.v_pooling = v_pooling
        self.bidirectional = bidirectional
        self.pos_encoder = PositionalEncoding(self.d_tf, dropout) if pos_encoder else None
        self.pooling = pooling or torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.tf = self.create_tf_model(**tf_kwargs)
        self.tf_ban = self.create_tf_model(**tf_ban_kwargs) if tf_ban_kwargs else None
        self.d_tf = self.tf.d_tf
        if self.tf_ban:
            self.d_tf += self.tf_ban.d_tf
        self.d_final = self._calc_d_final(self.d_tf)
        self._create_final(**final_kwargs)
        self._create_final_2(**{**final_kwargs, **final_2_kwargs})
        self._create_heads(**head_kwargs)

    def create_tf_model(self, **kwargs):
        return TransformerModel(
            embedding=self.embedding,
            bidirectional=self.bidirectional,
            pos_encoder=self.pos_encoder,
            pooling=self.pooling,
            **kwargs
        )
    
    def _calc_d_final(self, d_final):
        d_final = (2 if self.bidirectional == "concat" else 1) * d_final
        d_final = (1 if (
            self.pooling and not isinstance(self.pooling, torch.nn.Flatten)
        ) else 5) * d_final
        return d_final

    def _create_final(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        self.final = create_mlp_stack(self.d_final + 2, d_hid, self.d_final, n_layers, activation=activation, bias=bias, dropout=dropout)

    def _create_final_2(self, d_hid, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1):
        d_final_2 = (2*self.d_final) if self.final_2_mode == "double" else self.d_final
        self.final_2 = create_mlp_stack(d_final_2, d_hid, self.d_final, n_layers, activation=activation, bias=bias, dropout=dropout)

    def _create_heads(self, d_hid=0, n_layers=1, n_heads=1, d_heads=[120, 1], activation=torch.nn.ReLU, activations_final=[nn.Softmax, nn.Tanh], bias=True, dropout=0.1):
        d_hid = d_hid or self.d_final
        self.heads = [
            [
                nn.Sequential(*[
                    create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
                    nn.Sequential(*[
                        nn.Dropout(dropout),
                        nn.Linear(self.d_final, d_head, bias=bias),
                        activation_final()
                    ])
                ]) for d_head, activation_final in zip(d_heads, activations_final)
            ] for i in range(n_heads)
        ]

    def forward(self, left_bans, left_picks, right_bans, right_picks, count=2, next_count=2, legal_mask=None):

        tgt = self.tf(
            left_picks, right_picks
        )

        if self.tf_ban:
            tgt_ban = self.tf_ban(
                left_bans, right_bans
            )
            tgt = torch.cat([tgt_ban, tgt], dim=-1)

        tgt_0 = tgt

        try:
            tgt = self.final(torch.cat(
                [
                    tgt, 
                    torch.full([*tgt.shape[:-1], 1], count-1), 
                    torch.full([*tgt.shape[:-1], 1], next_count-1)
                ], 
                dim=-1
            ))
        except RuntimeError as ex:
            print(tgt.shape, self.d_final)
            raise
        #output = self.head(tgt)
        pi, v = [f(tgt) for f in self.heads[0]]
        
        if count > 1:
            if self.final_2_mode == "double":
                tgt_2 = self.final_2(torch.cat([tgt, tgt_0], dim=-1))
            else:
                tgt_2 = self.final_2(tgt)
            pi_2, v_2 = [f(tgt) for f in self.heads[-1]]
            if self.v_pooling == "1":
                pass
            elif self.v_pooling == "2":
                v = v_2
            else:
                v = torch.stack([v, v_2])
                if self.v_pooling in ("avg", "average", "mean"):
                    v = torch.mean(v, dim=0)
                elif self.v_pooling == "max":
                    v = torch.max(v, dim=0)
        else:
            pi_2 = torch.zeros(pi.shape)

        pis = (pi, pi_2)
        pi = torch.cat(pis, dim=-1)
        if legal_mask is not None and not self.training:
            pi = pi * legal_mask
        return pi, v
    
    def summary(self, batch_size=64, team_size=5, dtype=torch.int):
        return summary(
            self, 
            [(batch_size, team_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )

