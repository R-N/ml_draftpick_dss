from .util import sig_to_tanh_range, tanh_to_sig_range, split_dim
import torch
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
import copy
import time



class NegativeSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sig_to_tanh_range(torch.sigmoid(x))

class PositiveTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh_to_sig_range(torch.tanh(x))

class NegativeBCELoss(torch.nn.BCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        input = tanh_to_sig_range(input)
        target = tanh_to_sig_range(input)
        return super().forward(input, target)

class GlobalPooling1D(torch.nn.Module):
    def __init__(self, f=torch.mean, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    def forward(self, x):
        return self.f(x, dim=[-2])

class ResultPredictorModel(nn.Module):

    def __init__(self, 
        d_model, nhead, d_hid,
        nlayers, d_final,
        embedder=None, dropout=0.2,
        pooling=GlobalPooling1D
    ):
        super().__init__()
        if embedder:
            d_model = embedder.dim
        else:
            embedder = nn.Identity()
        self.model_type = 'Transformer'
        #self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.d_model = d_model
        self.encoder = embedder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.pooling = pooling()
        self.decoder = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            for i in range(d_final)
        ])
        self.victory_decoder = nn.Sequential(*[
            nn.Linear(d_model, 1),
            nn.Tanh()
        ])
        self.score_decoder = nn.Sequential(*[
            nn.Linear(d_model, 1),
            NegativeSigmoid()
        ])
        self.duration_decoder = nn.Sequential(*[
            nn.Linear(d_model, 1),
            NegativeSigmoid()
        ])

        #self.init_weights()
    """
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    """
    def forward(self, src, tgt):
        src = self.encoder(src)# * math.sqrt(self.d_model)
        #src = self.pos_encoder(src)
        
        memory = self.transformer_encoder(src)#, src_mask)
        tgt = self.transformer_decoder(tgt, memory)
        tgt = self.pooling(tgt)
        tgt = self.decoder(tgt)
        victory = self.victory_decoder(tgt)
        score = self.score_decoder(tgt)
        duration = self.duration_decoder(tgt)
        output = victory, score, duration
        #output = torch.cat(output, dim=-1)
        return output
    
class ResultPredictor:
    def __init__(
        self,
        d_model,
        d_hid=128,
        nlayers=2,
        nhead=2,
        d_final=2,
        model_embedder=None,
        dropout=0.2,
        device=None
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResultPredictorModel(d_model, nhead, d_hid, nlayers, d_final, embedder=model_embedder, dropout=dropout).to(device)
        self.epoch = 0
        self.training_prepared = False

    def prepare_training(
            self,
            train_loader,
            val_loader=None,
            victory_crit=NegativeBCELoss,
            norm_crit=torch.nn.MSELoss,
            lr=1e-3,
            optimizer=torch.optim.SGD
        ):
        self.victory_crit = victory_crit()
        self.norm_crit = norm_crit()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model.train()
        self.training_prepared = True

    def train(self):
        assert self.training_prepared
        self.model.train()  # turn on train mode
        total_victory_loss, total_score_loss, total_duration_loss, total_loss = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        batch_count = 0
        for i, batch in enumerate(self.train_loader):
            left, right, targets = batch
            victory_true, score_true, duration_true = split_dim(targets)
            victory_pred, score_pred, duration_pred = self.model(left, right)
            #victory_pred, norms_pred = preds[..., :1], preds[..., 1:]
            
            victory_loss = self.victory_crit(victory_pred, victory_true)
            score_loss = self.norm_crit(score_pred, score_true)
            duration_loss = self.norm_crit(duration_pred, duration_true)
            loss = victory_loss + score_loss + duration_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_victory_loss += victory_loss.item()
            total_score_loss += score_loss.item()
            total_duration_loss += duration_loss.item()
            total_loss += loss.item()
            batch_count += 1
    
        lr = self.scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / batch_count
        print(f'| epoch {self.epoch:3d} | step {i:5d} | '
            f'lr {lr} | ms/batch {ms_per_batch:5.2f} | ')
        self.epoch += 1
        return total_victory_loss / batch_count, total_score_loss / batch_count, total_duration_loss / batch_count, total_loss / batch_count