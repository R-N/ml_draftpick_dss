from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR, ReduceLROnPlateau as _ReduceLROnPlateau
import optuna

class ReduceLROnPlateau(_ReduceLROnPlateau):
    def __init__(self, *args, factor=0.1, patience=10, cooldown=2, min_lr=1e-7, raise_ex=True, **kwargs):
        super().__init__(*args, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr, **kwargs)
        self.raise_ex = raise_ex

    def _reduce_lr(self, epoch):
        updated = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                updated = True
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
        log = "Learning rate stuck"
        if not updated:
            print(log)
            if self.raise_ex:
                raise optuna.TrialPruned(log)

class OneCycleLR:
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, div_factor=25, autodecay=0.5):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.div_factor = max(25, div_factor)
        print("max_lr", self.max_lr)
        print("div_factor", self.div_factor)
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = epochs
        self.epochs = 0
        self.scheduler = None
        self.autodecay = autodecay
        self.create()

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def update_max_lr(self, max_lr, initial_lr=None, div_factor=None):
        if div_factor:
            self.div_factor = div_factor
        elif initial_lr:
            self.div_factor = max_lr / initial_lr
        elif self.initial_lr < max_lr:
            self.div_factor = max_lr / self.initial_lr
            self.div_factor = max(self.div_factor, 25)
        else:
            self.div_factor = 25
        self.max_lr = max_lr
        print("max_lr", self.max_lr)
        print("div_factor", self.div_factor)

    def create(self, last_epoch=-1):
        self.scheduler = _OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.max_epochs,
            last_epoch=last_epoch
        )
        return self.scheduler

    @property
    def initial_lr(self):
        return self.max_lr / self.div_factor

    @property
    def state_dict(self):
        return self.scheduler.state_dict

    def reset(self):
        """
        self.scheduler.last_epoch = -1
        self.scheduler.step()
        """
        if self.autodecay:
            self.update_max_lr(
                self.initial_lr * (
                    self.div_factor ** self.autodecay
                )
            )
        self.create()
        self.epochs = 0

    def step(self, *args, **kwargs):
        ret = self.scheduler.step()
        self.epochs += 1
        if self.epochs >= self.max_epochs:
            self.reset()
        return ret
