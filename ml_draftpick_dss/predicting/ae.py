import torch
from torchinfo import summary
import time
from ml_draftpick_dss.predicting.checkpoint import CheckpointManager, init_metrics
from ml_draftpick_dss.predicting.logging import TrainingLogger

class DiceLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):  
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def CosineLoss(*args, **kwargs):
    _CosineLoss = torch.nn.CosineEmbeddingLoss(*args, **kwargs)
    def __CosineLoss(x, y):
        return _CosineLoss(x, y, torch.ones([x.shape[0]]))
    return __CosineLoss

class HeroAEModel(torch.nn.Module):
    def __init__(self, dims, d_encoder=[128, 64, 32], d_decoder=[32, 64, 128], dropout=0, bias=True, activation=torch.nn.ReLU):
        super().__init__()
        self.d_input = sum(dims)
        assert len(d_encoder) > 0, "encoder must be provided"
        assert len(d_decoder) > 0, "decoder must be provided"
        assert d_encoder[-1] == d_decoder[0], "encoder and decoder dim mismatch"
        self.encoder = torch.nn.Sequential(*[
            torch.nn.Linear(self.d_input, d_encoder[0], bias=bias),
            activation(),
            *[
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(a, b, bias=bias),
                    activation(),
                ) for a, b in zip(d_encoder, d_encoder[1:])
            ]
        ]) 
        self.decoder = torch.nn.Sequential(*[
            *[
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(a, b, bias=bias),
                    activation(),
                ) for a, b in zip(d_decoder, d_decoder[1:])
            ]
            #torch.nn.Linear(d_decoder[-1], self.d_input, bias=bias),
            #activation(),
        ]) 
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.Linear(d_decoder[-1], d, bias=bias),
                torch.nn.Softmax(-1)
            ])
            for d in dims
        ])
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        splitted = [head(decoded) for head in self.heads]
        combined = torch.cat(splitted, dim=-1)
        return combined
    
    def summary(self, batch_size=32, dim=0):
        dim = dim or self.d_model
        return summary(
            self, 
            [(batch_size, dim)], 
            dtypes=[torch.float]
        )
    
class HeroAE:
    def __init__(
        self,
        dims,
        slices,
        *args,
        device=None,
        grad_clipping=0,
        **kwargs
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HeroAEModel(dims, *args, **kwargs).to(device)
        self.epoch = 0
        self.slices = slices
        self.training_prepared = False
        self.grad_clipping = grad_clipping

    def prepare_training(
        self,
        train_loader,
        val_loader=None,
        crit=torch.nn.CrossEntropyLoss(),
        lr=1e-3,
        optimizer=torch.optim.Adam,
        metrics=["loss"],
        checkpoint_dir="checkpoints",
        log_dir="logs",
    ):
        self.crit = crit
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model.train()

        self.metrics = metrics
        self.best_metrics = init_metrics(self.metrics)

        self.prepare_checkpoint(checkpoint_dir)
        self.prepare_logging(log_dir)

        self.training_prepared = True

    def prepare_checkpoint(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_managers = {m: CheckpointManager(self, m, self.checkpoint_dir) for m in self.metrics}

    def prepare_logging(self, log_dir="logs"):
        self.log_dir = log_dir
        self.logger = TrainingLogger(log_dir)

    def log_scalar(self, *args, **kwargs):
        self.logger.log_scalar(*args, **kwargs)

    def load_checkpoint(self, checkpoint="loss"):
        assert self.checkpoint_managers
        cm = self.checkpoint_managers[checkpoint]
        cm.load_checkpoint()

    def save_checkpoint(self, checkpoint):
        assert self.checkpoint_managers
        cm = self.checkpoint_managers[checkpoint]
        cm.save_checkpoint()

    def set_lr(self, lr):
        for g in self.optim.param_groups:
            g['lr'] = lr

    def train(self):
        assert self.training_prepared
        self.model.train()  # turn on train mode
        losses = {
            "loss": 0
        }
        start_time = time.time()

        batch_count = 0
        for i, batch in enumerate(self.train_loader):
            batch = batch[0]
            preds = self.model(batch)
            
            loss = [self.crit(preds[..., start:end], batch[..., start:end]) for start, end in self.slices]
            loss = torch.sum(torch.stack(loss))

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.optimizer.step()

            losses["loss"] += loss.item()
            batch_count += 1

        losses = {k: v/batch_count for k, v in losses.items()}
        cur_metrics = {
            "epoch": self.epoch,
            **losses,
        }
    
        lr = self.scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / batch_count
        print(f'| epoch {self.epoch:3d} | step {i:5d} | '
            f'lr {lr} | ms/batch {ms_per_batch:5.2f} | ')
        self.scheduler.step()
        
        for m, v in cur_metrics.items():
            self.log_scalar(m, v, self.epoch)

        new_best_metrics = []
        for m in self.metrics:
            cm = self.checkpoint_managers[m]
            ret = cm.check_metric(cur_metrics)
            if ret:
                new_best_metrics.append(ret)

        self.epoch += 1
        return self.epoch, cur_metrics, new_best_metrics
    
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)