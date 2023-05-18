from ...util import mkdir
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier, metrics
import os

METRICS = [
    "loss",
    "accuracy",
    "auc",
    "f1_score",
]
VAL_METRICS = [f"val_{m}" for m in METRICS]
TARGETS = ["victory", "score", "duration"]

class ResultPredictor:
    def __init__(
        self,
        *args,
        epochs=1000,
        lr=0.1,
        bin_crit=metrics.CrossEntropy(),
        metric=metrics.Accuracy(),
        random_seed=42,
        od_wait=50,
        model=CatBoostClassifier,
        **kwargs
    ):
        self.lr = lr
        self.epochs = epochs
        self.epoch = 0
        self.od_wait = od_wait
        self.model = model(
            *args, 
            iterations=epochs,
            learning_rate=lr,
            loss_function=bin_crit,
            eval_metric=metric,
            random_seed=random_seed,
            od_wait=od_wait,
            use_best_model=True,
            od_type="Iter",
            logging_level="Silent",
            **kwargs
        )
        self.training_prepared = False

    def prepare_training(
        self,
        train_loader,
        val_loader=None,
        checkpoint_dir="checkpoints",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.prepare_checkpoint(checkpoint_dir)

        self.training_prepared = True

    def prepare_checkpoint(self, checkpoint_dir="checkpoints"):
        mkdir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def load_model(self, file_name="best.dump"):
        self.model.load_model(os.path.join(self.checkpoint_dir, file_name))

    def save_model(self, file_name="best.dump"):
        self.model.save_model(os.path.join(self.checkpoint_dir, file_name))

    def get_lr(self):
        return self.lr

    def train(self, val=False, val_loader=None, autosave=True, true_threshold=0.5):
        assert self.training_prepared
        val_loader = self.val_loader if val_loader is None else val_loader
        if val:
            assert val_loader is not None, "Please provide validation dataloader"
        start_time = time.time()

        bin_true = []
        bin_pred = []
        loader = val_loader if val else self.train_loader

        if not val:
            self.model.fit(
                loader,
                eval_set=val_loader,
                logging_level="Verbose",
                plot=True
            )
            if self.od_wait:
                self.epoch = self.model.get_best_iteration() + self.od_wait
            else:
                self.epoch = self.epochs

        bin_true = loader.y > true_threshold
        bin_pred = self.model.predict(loader) > true_threshold

        cm = confusion_matrix(bin_true, bin_pred)
        cm_labels = ["tn", "fp", "fn", "tp"]

        loss = np.mean(self.model.eval_metrics(loader, [metrics.CrossEntropy()])['CrossEntropy'])

        cur_metrics = {
            "epoch": self.epoch,
            "loss": loss,
            "victory_loss": loss,
            "accuracy": accuracy_score(bin_true, bin_pred),
            "auc": roc_auc_score(bin_true, bin_pred),
            "f1_score": f1_score(bin_true, bin_pred),
            **{cm_labels[i]: x for i, x in enumerate(cm.ravel())}
        }
    
        if val:
            cur_metrics = {f"val_{k}": v for k, v in cur_metrics.items()}
        else:
            lr = self.get_lr()
            ms = (time.time() - start_time) * 1000
            print(f'| epoch {self.epoch:3d} | '
                f'lr {lr} | ms {ms:5.2f} | ')
            
        if autosave:
            self.save_model()

        return self.epoch, cur_metrics
    
    def feature_importance(self, columns):
        feature_importances = self.model.get_feature_importance(self.train_loader)
        feature_names = [f"{l}_{c}" for c in columns for l in ("left", "right")]
        assert (len(feature_importances) == len(feature_names))
        return {n: s for s, n in sorted(zip(feature_importances, feature_names), reverse=True)}

    def predict(self, data):
        return self.model.predict(data)

    def predict_prob(self, data):
        return self.model.predict_proba(data)
