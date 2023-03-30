from .util import sig_to_tanh_range, tanh_to_sig_range, split_dim
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from torchinfo import summary


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

MEAN = torch.mean
PROD = torch.prod
SUM = torch.sum
MAX = torch.max

class GlobalPooling1D(torch.nn.Module):
    def __init__(self, f=MEAN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    def forward(self, x):
        return self.f(x, dim=-2)

class ResultPredictorModel(nn.Module):

    def __init__(self, 
        d_model, nhead, d_hid,
        nlayers, d_final,
        embedder=None, dropout=0.2,
        pooling=GlobalPooling1D,
        act_final=nn.Tanh
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
        self.decoder = nn.Sequential(
            *[
                nn.Linear(d_model, d_hid),
                act_final(),
                #nn.Dropout(dropout)
            ],
            *[
                nn.Sequential(*[
                    nn.Linear(d_hid, d_hid),
                    act_final(),
                    nn.Dropout(dropout)
                ])
                for i in range(max(0, d_final))
            ],
            *[
                nn.Linear(d_hid, d_model),
                act_final(),
                #nn.Dropout(dropout)
            ],
        )
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

        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        for l0 in [
            self.decoder,
            self.victory_decoder,
            self.score_decoder,
            self.duration_decoder
        ]:
            if not isinstance(l0, nn.Sequential):
                l0 = [l0]
            for l1 in l0:
                l1.bias.data.zero_()
                l1.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt):
        src = self.encoder(src)# * math.sqrt(self.d_model)
        #src = self.pos_encoder(src)
        
        memory = self.transformer_encoder(src)#, src_mask)
        tgt = self.encoder(tgt)# * math.sqrt(self.d_model)
        #tgt = self.pos_encoder(tgt)
        tgt = self.transformer_decoder(tgt, memory)
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
        device=None,
        log_dir="logs"
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResultPredictorModel(d_model, nhead, d_hid, nlayers, d_final, embedder=model_embedder, dropout=dropout).to(device)
        self.epoch = 0
        self.training_prepared = False
        self.log_dir = log_dir
        self.file_writers = None

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

    def prepare_logging(self):
        self.file_writers = {
            "train": tf.summary.create_file_writer(self.log_dir + f"/train"),
            "val": tf.summary.create_file_writer(self.log_dir + f"/val"),
        }

    def train(self):
        assert self.training_prepared
        self.model.train()  # turn on train mode
        losses = {
            "total_victory_loss": 0, 
            "total_score_loss": 0, 
            "total_duration_loss": 0, 
            "total_loss": 0
        }
        start_time = time.time()

        batch_count = 0
        bin_true = []
        bin_pred = []
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

            losses["total_victory_loss"] += victory_loss.item()
            losses["total_score_loss"] += score_loss.item()
            losses["total_duration_loss"] += duration_loss.item()
            losses["total_loss"] += loss.item()
            batch_count += 1
            bin_true.extend(list(torch.squeeze(victory_true, dim=-1) > 0))
            bin_pred.extend(list(torch.squeeze(victory_pred, dim=-1) > 0))

        bin_true, bin_pred = np.array(bin_true).astype(int), np.array(bin_pred).astype(int)
        cm = confusion_matrix(bin_true, bin_pred)
        cm_labels = ["tn", "fp", "fn", "tp"]

        losses = {k: v/batch_count for k, v in losses.items()}
        cur_metrics = {
            "epoch": self.epoch,
            **losses,
            "accuracy": accuracy_score(bin_true, bin_pred),
            "auc": roc_auc_score(bin_true, bin_pred),
            "f1": f1_score(bin_true, bin_pred),
            **{cm_labels[i]: x for i, x in enumerate(cm.ravel())}
        }
    
        lr = self.scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / batch_count
        print(f'| epoch {self.epoch:3d} | step {i:5d} | '
            f'lr {lr} | ms/batch {ms_per_batch:5.2f} | ')
        self.epoch += 1
        return cur_metrics
    
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)