
import torch
import os
import json
from ..util import mkdir

METRICS = [
    "victory_loss",
    "score_loss",
    "duration_loss",
    "loss",
    "epoch",
    "accuracy",
    "auc",
    "f1_score",
]
VAL_METRICS = [f"val_{m}" for m in METRICS]

def init_metrics(metrics=METRICS):
    return {m: (100 if "loss" in m else 0) for m in metrics}

class CheckpointManager:
    def __init__(self, model, metric, checkpoint_dir="checkpoints"):
        assert metric in (METRICS+VAL_METRICS), f"Invalid metric: {metric}"
        self.model = model
        self.metric = metric
        checkpoint_dir = os.path.join(checkpoint_dir, metric)
        checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path
        self.metrics_path = metrics_path
        mkdir(checkpoint_dir)
        self.best_metrics = init_metrics()
        self.best_metrics = self.model.best_metrics.copy()
        self.load_best_metrics()

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.epoch = checkpoint['epoch']
            self.model.best_metrics = checkpoint["best_metrics"]
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.load_best_metrics(True)
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as ex:
            print(ex)

    def save_checkpoint(self):
        torch.save({
            'epoch': self.model.epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'scheduler_state_dict': self.model.scheduler.state_dict(),
            'best_metrics': self.model.best_metrics,
        }, self.checkpoint_path)
        self.save_best_metrics()

    def check_metric(self, cur_metrics, save=True):
        if not self.best_metrics:
            self.best_metrics = cur_metrics
            self.save_best_metrics()
            return
        if self.metric not in cur_metrics:
            return
        
        m = self.metric
        cur_val, best_val = cur_metrics[m], self.best_metrics[m]
        ret = None
        if "loss" in m:
            cur_val, best_val = -cur_val, -best_val
        if cur_val >= best_val:
            cur_val, best_val = cur_metrics[m], self.best_metrics[m]
            ret = (m, best_val, cur_val)
            self.model.best_metrics[m] = cur_val
            self.best_metrics = self.model.best_metrics.copy()
            if save:
                self.save_checkpoint()
            else:
                self.save_best_metrics()
        return ret

    def save_best_metrics(self):
        with open(self.metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=4)

    def load_best_metrics(self, model=False):
        try:
            with open(self.metrics_path, 'r') as f:
                best_metrics = json.load(f)
            self.best_metrics = best_metrics
            if model:
                self.model.best_metrics = best_metrics.copy()
            return best_metrics
        except Exception as ex:
            print(ex)