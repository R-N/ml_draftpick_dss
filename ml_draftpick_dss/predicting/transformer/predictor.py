from ..util import split_dim
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from .model import ResultPredictorModel
from ..checkpoint import CheckpointManager, METRICS, VAL_METRICS, init_metrics
from ..logging import TrainingLogger
from ..lr_finder import LRFinder
from ..early_stopping import EarlyStopping
from ..scheduler import OneCycleLR, ReduceLROnPlateau

TARGETS = ["victory", "score", "duration"]

def scale_loss(x):
    return (1.0/(2*torch.var(x)))

def extra_loss(losses):
    return torch.log(torch.prod(torch.stack([torch.std(loss) for loss in losses])))

class ResultPredictor:
    def __init__(
        self,
        *args,
        device=None,
        model=ResultPredictorModel,
        reduce_loss=torch.mean,
        scale_loss=lambda x: 1,
        extra_loss=lambda losses: 0,
        **kwargs
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(*args, **kwargs).to(device)
        self.epoch = 0
        self.training_prepared = False
        self.reduce_loss = reduce_loss
        self.scale_loss = scale_loss
        self.extra_loss = extra_loss

    

    def prepare_training(
        self,
        train_loader,
        val_loader=None,
        bin_crit=torch.nn.BCELoss(reduction="none"),
        norm_crit=None,
        lr=1e-2,
        optimizer=torch.optim.Adam,
        grad_clipping=0,
        metrics=METRICS+VAL_METRICS,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        scheduler_type="onecycle",
        scheduler_kwargs={}
    ):
        self.bin_crit = bin_crit
        self.norm_crit = norm_crit
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_type = scheduler_type
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

    def create_scheduler(self):
        print("creating scheduler of type", self.scheduler_type)
        if self.scheduler_type == None:
            self.scheduler = None
            return
        if self.scheduler_type == "plateau":
            scheduler_f = self.create_plateau_scheduler
        elif self.scheduler_type == "onecycle":
            scheduler_f = self.create_onecycle_scheduler
        else:
            raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")
        self.scheduler = scheduler_f(**self.scheduler_kwargs)

    def remove_scheduler(self):
        self.scheduler_type = None
        self.scheduler = None

    def create_plateau_scheduler(self, patience=10, cooldown=2, threshold=1e-4, min_lr=0, eps=1e-8, verbose=True, **kwargs):
        return ReduceLROnPlateau(
            self.optimizer, 
            patience=patience,
            cooldown=cooldown,
            threshold=threshold,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
    def prepare_checkpoint(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_managers = {m: CheckpointManager(self, m, self.checkpoint_dir) for m in self.metrics} if checkpoint_dir else {}

    def prepare_logging(self, log_dir="logs"):
        self.log_dir = log_dir
        self.logger = TrainingLogger(log_dir) if log_dir else {}

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

    def set_lr(self, lr, **kwargs):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.create_scheduler(**kwargs)

    def get_lr(self, index=0):
        return self.optimizer.param_groups[index]["lr"]

    def train(self, val=False, val_loader=None, autosave=True, true_threshold=0.5):
        assert self.training_prepared
        if val:
            self.model.eval()
            val_loader = self.val_loader if val_loader is None else val_loader
            assert val_loader is not None, "Please provide validation dataloader"
        else:
            self.model.train()  # turn on train mode
        losses = {
            **{f"{t}_loss": 0 for t in TARGETS},
            #**{f"scaled_{t}_loss": 0 for t in TARGETS},
            #"scaled_loss": 0,
            #"extra_loss": 0,
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
            left, right, targets, weights = batch

            _trues = split_dim(targets)
            _preds = self.model(left, right)

            victory_true, score_true, duration_true = _trues
            victory_pred, score_pred, duration_pred = _preds
            
            victory_true = victory_true * (1.0 - true_threshold) + true_threshold
            
            victory_loss = self.bin_crit(victory_pred, victory_true)
            score_loss = 0
            duration_loss = 0
            if self.norm_crit:
                score_loss = self.norm_crit(score_pred, score_true)
                duration_loss = self.norm_crit(duration_pred, duration_true)
                raw_losses = (victory_loss, score_loss, duration_loss)
            else:
                raw_losses = (victory_loss,)

            weighted_losses = [weights * raw_losses[i] for i in range(len(raw_losses))]
            reduced_losses = torch.stack([self.reduce_loss(x) for x in weighted_losses])

            """
            scaled_losses = torch.stack([self.scale_loss(x) * y for x, y in zip(raw_losses, reduced_losses)])
            scaled_loss = torch.sum(scaled_losses)
            extra_loss = self.extra_loss(raw_losses)
            loss = scaled_loss + extra_loss
            """
            loss = torch.sum(reduced_losses)

            if not val:
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                self.optimizer.step()

            new_losses = {
                **{f"{t}_loss": l for t, l in zip(TARGETS, reduced_losses)},
                # **{f"scaled_{t}_loss": l for t, l in zip(TARGETS, scaled_losses)},
                #"scaled_loss": scaled_loss,
                #"extra_loss": extra_loss, 
                "loss": loss
            }
            for k, v in new_losses.items():
                losses[k] += v.item() if torch.is_tensor(v) else v
            batch_count += 1
            #min_victory_pred = min(min_victory_pred, torch.min(victory_pred).item())
            #max_victory_pred = max(max_victory_pred, torch.max(victory_pred).item())

            squeezed_pred = torch.squeeze(victory_pred, dim=-1)
            bin_true.extend(list(torch.squeeze(victory_true, dim=-1) > true_threshold))
            bin_pred.extend(list(squeezed_pred > true_threshold))
            victory_preds = torch.cat([victory_preds, squeezed_pred], dim=-1)

        bin_true, bin_pred = np.array(bin_true).astype(int), np.array(bin_pred).astype(int)
        cm = confusion_matrix(bin_true, bin_pred)
        cm_labels = ["tn", "fp", "fn", "tp"]

        """
        min_victory_pred = torch.min(victory_preds).item()
        max_victory_pred = torch.max(victory_preds).item()
        mean_victory_pred = torch.mean(victory_preds).item()
        """

        losses = {k: v/batch_count for k, v in losses.items()}
        cur_metrics = {
            "epoch": self.epoch,
            **losses,
            "accuracy": accuracy_score(bin_true, bin_pred),
            "auc": roc_auc_score(bin_true, bin_pred),
            "f1_score": f1_score(bin_true, bin_pred),
            #"mean_victory_pred": mean_victory_pred,
            #"min_victory_pred": min_victory_pred,
            #"max_victory_pred": max_victory_pred,
            **{cm_labels[i]: x for i, x in enumerate(cm.ravel())}
        }
    
        if val:
            cur_metrics = {f"val_{k}": v for k, v in cur_metrics.items()}
        else:
            lr = self.get_lr()
            ms_per_batch = (time.time() - start_time) * 1000 / batch_count
            print(f'| epoch {self.epoch:3d} | step {i:5d} | '
                f'lr {lr} | ms/batch {ms_per_batch:5.2f} | ')
            if self.scheduler:
                self.scheduler.step(loss.item())
        
        if self.logger:
            for m, v in cur_metrics.items():
                self.log_scalar(m, v, self.epoch)

        new_best_metrics = []
        if self.checkpoint_managers:
            for m in self.metrics:
                cm = self.checkpoint_managers[m]
                ret = cm.check_metric(cur_metrics, save=autosave)
                if ret:
                    new_best_metrics.append(ret)
        return self.epoch, cur_metrics, new_best_metrics
    
    def inc_epoch(self):
        self.epoch += 1
    
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def find_lr(self, loss_fn=None, min_epoch=50, **kwargs):
        def objective(scheduler):
            return self.train(
                loss_fn=loss_fn,
                scheduler=scheduler,
                clip_grad_norm=False
            )[0].item()

        lr_finder = LRFinder(objective, self.models, self.optimizer)
        lr_finder.range_test(num_iter=int(0.5 * min_epoch), **kwargs)
        lr_finder.reset_state()
        return lr_finder.result
    
    def create_early_stopping_1(self, wait=50, max_epoch=200, **kwargs):
        early_stopping_1 = EarlyStopping(
            self.model,
            log_dir=self.log_dir,
            label=self.model.name,
            wait=wait,
            max_epoch=max_epoch,
            **kwargs
        )
        return early_stopping_1
    
    def create_early_stopping_2(self, early_stopping_1, **kwargs):
        early_stopping_2 = EarlyStopping(
            self.model,
            log_dir=self.log_dir,
            label=self.model.name + "a",
            interval_mode=early_stopping_1.interval_mode,
            wait=0,
            max_epoch=early_stopping_1.best_epoch,
            update_state_mode=1,
            **kwargs
        )
        return early_stopping_2
    
    def create_onecycle_scheduler(self, steps, mul=10, epochs=50, **kwargs):
        return OneCycleLR(
            self.optimizer,
            max_lr=min(1.0, mul*self.get_lr()),
            steps_per_epoch=steps,
            epochs=epochs
        )
