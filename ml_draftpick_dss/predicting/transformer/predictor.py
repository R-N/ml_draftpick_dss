from .util import split_dim
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from .model import ResultPredictorModel
from ..checkpoint import CheckpointManager, METRICS, init_metrics
from ..logging import TrainingLogger


class ResultPredictor:
    def __init__(
        self,
        *args,
        device=None,
        **kwargs
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResultPredictorModel(*args, **kwargs).to(device)
        self.epoch = 0
        self.training_prepared = False

    def prepare_training(
        self,
        train_loader,
        val_loader=None,
        bin_crit=torch.nn.BCELoss(),
        norm_crit=torch.nn.MSELoss(reduction="mean"),
        lr=1e-3,
        optimizer=torch.optim.Adam,
        grad_clipping=0,
        metrics=METRICS,
        checkpoint_dir="checkpoints",
        log_dir="logs",
    ):
        self.bin_crit = bin_crit
        self.norm_crit = norm_crit
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.create_scheduler()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clipping = grad_clipping
        self.model.train()

        self.metrics = metrics
        self.best_metrics = init_metrics(self.metrics)

        self.prepare_checkpoint(checkpoint_dir)
        self.prepare_logging(log_dir)

        self.training_prepared = True

    def create_scheduler(self, step_size=3, gamma=0.95):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

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
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.create_scheduler()

    def train(self, val=False, val_loader=None):
        assert self.training_prepared
        if val:
            self.model.eval()
            val_loader = self.val_loader if val_loader is None else val_loader
            assert val_loader is not None, "Please provide validation dataloader"
        else:
            self.model.train()  # turn on train mode
        losses = {
            "victory_loss": 0, 
            "score_loss": 0, 
            "duration_loss": 0, 
            "loss": 0
        }
        start_time = time.time()

        batch_count = 0
        bin_true = []
        bin_pred = []
        #min_victory_pred, max_victory_pred = 2, -2
        victory_preds = torch.Tensor([])
        loader = val_loader if val else self.train_loader
        for i, batch in enumerate(loader):
            left, right, targets = batch
            victory_true, score_true, duration_true = split_dim(targets)
            victory_pred, score_pred, duration_pred = self.model(left, right)
            #victory_pred, norms_pred = preds[..., :1], preds[..., 1:]
            
            victory_loss = self.norm_crit(victory_pred, victory_true)
            score_loss = self.norm_crit(score_pred, score_true)
            duration_loss = self.norm_crit(duration_pred, duration_true)
            loss = victory_loss + score_loss + duration_loss

            if not val:
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                self.optimizer.step()

            losses["victory_loss"] += victory_loss.item()
            losses["score_loss"] += score_loss.item()
            losses["duration_loss"] += duration_loss.item()
            losses["loss"] += loss.item()
            batch_count += 1
            #min_victory_pred = min(min_victory_pred, torch.min(victory_pred).item())
            #max_victory_pred = max(max_victory_pred, torch.max(victory_pred).item())

            squeezed_pred = torch.squeeze(victory_pred, dim=-1)
            bin_true.extend(list(torch.squeeze(victory_true, dim=-1) > 0))
            bin_pred.extend(list(squeezed_pred > 0))
            victory_preds = torch.cat([victory_preds, squeezed_pred], dim=-1)

        bin_true, bin_pred = np.array(bin_true).astype(int), np.array(bin_pred).astype(int)
        cm = confusion_matrix(bin_true, bin_pred)
        cm_labels = ["tn", "fp", "fn", "tp"]

        min_victory_pred = torch.min(victory_preds).item()
        max_victory_pred = torch.max(victory_preds).item()
        mean_victory_pred = torch.mean(victory_preds).item()

        losses = {k: v/batch_count for k, v in losses.items()}
        cur_metrics = {
            "epoch": self.epoch,
            **losses,
            "accuracy": accuracy_score(bin_true, bin_pred),
            "auc": roc_auc_score(bin_true, bin_pred),
            "f1_score": f1_score(bin_true, bin_pred),
            "mean_victory_pred": mean_victory_pred,
            "min_victory_pred": min_victory_pred,
            "max_victory_pred": max_victory_pred,
            **{cm_labels[i]: x for i, x in enumerate(cm.ravel())}
        }
    
        if val:
            cur_metrics = {f"val_{k}": v for k, v in cur_metrics.items()}
        else:
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
        if not val:
            self.epoch += 1
        return self.epoch, cur_metrics, new_best_metrics
    
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)