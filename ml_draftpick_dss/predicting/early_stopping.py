from .util import progressive_smooth, calculate_prediction_interval
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import optuna


class EarlyStopping:
    def __init__(
        self,
        model,
        wait=50, wait_train_below_val=0,
        rise_patience=13, still_patience=13, both_patience=38,
        interval_percent=0.05,
        history_length=10,
        smoothing=0.05,
        interval_mode=1,
        max_epoch=100,
        max_nan=None,
        rise_forgiveness=0.6,
        still_forgiveness=0.6,
        both_forgiveness=0.66,
        decent_forgiveness_mul=0.6,
        small_forgiveness_mul=0.4,
        mini_forgiveness_mul=0.2,
        debug=1,
        log_dir=None,
        label=None,
        eps=1e-3,
        update_state_mode=2,
        raise_ex=True,
    ):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.model = model
        self.wait = wait
        self.wait_counter = 0
        self.wait_train_below_val = wait_train_below_val
        self.wait_train_below_val_counter = 0
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.both_patience = both_patience
        self.min_delta_val = 0
        self.min_delta_train = 0
        self.rise_counter = 0
        self.still_counter = 0
        self.both_counter = 0
        self.history_length = history_length or min(rise_patience, still_patience)
        self.half_history_length = int(self.history_length / 2)
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_loss_history_2 = []
        self.val_loss_history_2 = []
        self.best_val_loss = None
        self.best_train_loss = None
        self.best_val_loss_2 = None
        self.stopped = False
        self.smoothing = min(1.0, max(0, smoothing))
        self.interval_percent = interval_percent
        self.debug = debug
        self.interval_funcs = [
            self.calculate_interval_0,
            self.calculate_interval_1
        ]
        self.interval_mode = interval_mode

        self.rise_forgiveness = rise_forgiveness
        self.still_forgiveness = still_forgiveness
        self.both_forgiveness = both_forgiveness or (max(rise_patience, still_patience) / (both_patience - 1))
        self.decent_forgiveness_mul = decent_forgiveness_mul
        self.small_forgiveness_mul = small_forgiveness_mul
        self.mini_forgiveness_mul = mini_forgiveness_mul

        self.max_epoch = max_epoch
        self.log_dir = log_dir
        self.label = label
        self.epoch = 0
        self.active = False

        self.train_loss = None
        self.val_loss = None

        self.max_nan = max_nan or max(0, int(0.5 * (self.wait - self.history_length)))
        self.nan_counter = 0

        self.eps = eps
        self.update_state_mode = update_state_mode

        self.last_epoch = 0
        self.stop_reason = None

        self.raise_ex = raise_ex

        if self.log_dir is not None:
            assert self.label is not None
            self.min_high_writer = SummaryWriter(log_dir + "/min_high")
            self.min_low_writer = SummaryWriter(log_dir + "/min_low")

            self.mean_loss_writer = SummaryWriter(log_dir + "mean_loss")
            self.mean_loss_half_writer = SummaryWriter(log_dir + "mean_loss_half")
            self.min_high_mean_writer = SummaryWriter(log_dir + "min_high_mean")
            self.min_low_mean_writer = SummaryWriter(log_dir + "min_low_mean")

            self.loss_writer = SummaryWriter(log_dir + "/loss")
            self.best_loss_writer = SummaryWriter(log_dir + "/best_loss")
            self.best_loss_2_writer = SummaryWriter(log_dir + "/best_loss_2")

            self.still_writer = SummaryWriter(log_dir + "/still")
            self.rise_writer = SummaryWriter(log_dir + "/rise")
            self.both_writer = SummaryWriter(log_dir + "/both")

    def calculate_interval(self, val=None, history=None, *args, **kwargs):
        assert val is not None or history is not None
        if history is None:
            history = self.val_loss_history_2 if val else self.train_loss_history_2
        mid, delta = self.interval_funcs[self.interval_mode](history, *args, **kwargs)
        delta = max(delta, self.eps)
        return mid, delta

    def calculate_interval_0(self, history):
        min_val = min(history)
        max_val = max(history)
        delta = 0.5 * (1.0 - self.interval_percent) * (max_val - min_val)
        mid = 0.5 * (min_val + max_val)
        return mid, delta

    def calculate_interval_1(self, history):
        return calculate_prediction_interval(history, self.interval_percent)

    def log_stop(
        self,
        label, epoch,
        loss,
        min_delta=None, min_delta_2=None,
        best_loss=None, best_loss_2=None,
        mean_loss=None, mean_loss_half=None
    ):
        self.loss_writer.add_scalar(self.label + label, loss, global_step=epoch)
        self.loss_writer.flush()

        if mean_loss is not None:
            self.mean_loss_writer.add_scalar(self.label + label, mean_loss, global_step=epoch)
            self.mean_loss_writer.flush()

        if mean_loss_half is not None:
            self.mean_loss_half_writer.add_scalar(self.label + label, mean_loss_half, global_step=epoch)
            self.mean_loss_half_writer.flush()

        if mean_loss is not None and min_delta_2 is not None:
            self.min_high_mean_writer.add_scalar(self.label + label, mean_loss + min_delta_2, global_step=epoch)
            self.min_low_mean_writer.add_scalar(self.label + label, mean_loss - min_delta_2, global_step=epoch)
            self.min_high_mean_writer.flush()
            self.min_low_mean_writer.flush()

        if min_delta is not None:
            self.min_high_writer.add_scalar(self.label + label, best_loss + min_delta, global_step=epoch)
            self.min_low_writer.add_scalar(self.label + label, best_loss - min_delta, global_step=epoch)
            self.min_high_writer.flush()
            self.min_low_writer.flush()

        if best_loss is not None:
            self.best_loss_writer.add_scalar(self.label + label, best_loss, global_step=epoch)
            self.best_loss_writer.flush()

        if best_loss_2 is not None:
            self.best_loss_2_writer.add_scalar(self.label + label, best_loss_2, global_step=epoch)
            self.best_loss_2_writer.flush()

    def __call__(self, train_loss, val_loss=None, epoch=None):
        epoch = epoch if epoch is not None else self.epoch

        val_loss = train_loss if val_loss is None else val_loss
        val_loss_0, train_loss_0 = val_loss, train_loss

        mean_train_loss, min_delta_train_2 = self.calculate_interval(val=True)
        mean_val_loss, min_delta_val_2 = self.calculate_interval(val=True)
        if self.val_loss_history:
            mean_val_loss = sum(self.val_loss_history[-self.history_length:]) / min(self.history_length, len(self.val_loss_history))
            mean_val_loss_half = sum(self.val_loss_history[-self.half_history_length:]) / min(self.half_history_length, len(self.val_loss_history))
        else:
            mean_val_loss_half = mean_val_loss
        # min_delta_val_2 *= 0.75

        self.train_loss_history = [*self.train_loss_history, train_loss][-self.history_length:]
        self.val_loss_history = [*self.val_loss_history, val_loss][-self.history_length:]
        self.train_loss_history_2 = [*self.train_loss_history_2, train_loss][-self.history_length:]
        self.val_loss_history_2 = [*self.val_loss_history_2, val_loss][-self.history_length:]

        if self.val_loss is None:
            self.val_loss = val_loss
            self.train_loss = train_loss
        else:
            train_loss = progressive_smooth(self.train_loss, self.smoothing, train_loss_0)
            val_loss = progressive_smooth(self.val_loss, self.smoothing, val_loss_0)

        if self.wait_counter < self.wait:
            self.wait_counter += 1
        elif not self.active and val_loss < train_loss and self.wait_train_below_val_counter < self.wait_train_below_val:
            self.wait_train_below_val_counter += 1
        elif not self.active:
            self.active = True
            self.forgive_both(
                mul=self.small_forgiveness_mul,
                min_forgiveness=self.forgive_still(
                    self.small_forgiveness_mul
                ) + self.forgive_rise(
                    self.small_forgiveness_mul
                )
            )
            print(f"INFO: Early stopping active at epoch {epoch} after skipping {self.nan_counter}/{self.max_nan} NaN epochs and waiting {self.wait_train_below_val_counter}/{self.wait_train_below_val} epochs for train to get below val")

        if self.best_val_loss is None:
            self.update_best_val_2(val_loss)
            self.update_best_train(train_loss)
            self.update_best_val(val_loss)
            self.recalculate_delta_val()
            self.recalculate_delta_train()

        min_delta_val = self.min_delta_val
        min_delta_train = self.min_delta_train


        if self.log_dir:
            self.log_stop(
                label="val_stop", epoch=epoch,
                loss=val_loss, mean_loss=mean_val_loss, mean_loss_half=mean_val_loss_half,
                min_delta=self.min_delta_val, min_delta_2=min_delta_val_2,
                best_loss=self.best_val_loss, best_loss_2=self.best_val_loss_2
            )
            self.log_stop(
                label="train_stop", epoch=epoch,
                loss=train_loss,
                min_delta=self.min_delta_train,
                best_loss=self.best_train_loss
            )

        delta_val_loss = val_loss - self.best_val_loss
        delta_train_loss = train_loss - self.best_train_loss

        train_fall = delta_train_loss < -min_delta_train
        val_rise = delta_val_loss > min_delta_val
        val_still = abs(delta_val_loss) < min_delta_val
        val_fall = delta_val_loss < -min_delta_val
        val_fall_1 = val_loss < self.best_val_loss_2

        delta_train_loss_2 = train_loss - mean_train_loss
        train_still_2 = (delta_train_loss_2) < min_delta_train_2
        delta_val_loss_2 = val_loss - mean_val_loss
        val_fall_2 = delta_val_loss_2 < -min_delta_val_2
        val_still_2 = abs(delta_val_loss_2) < min_delta_val_2

        if self.wait_counter > 3:
            if not train_still_2:
                del self.train_loss_history[-1]
            if not val_still_2:
                del self.val_loss_history[-1]

        val_decrease = val_loss < self.val_loss and val_loss < mean_val_loss_half

        if train_fall:
            self.update_best_train(train_loss)
        if delta_train_loss < min_delta_train:
            self.recalculate_delta_train()

        if val_rise:
            rise_increment = 1
            still_increment = 0
            if val_still_2:
                still_increment = 0.85
                if val_decrease:
                    still_increment *= (1.0 - 0.15)
                    rise_increment *= (1.0 - 0.4)
                else:
                    rise_increment *= (1.0 - 0.2)
                self.still_counter += still_increment
            else:
                self.forgive_still(self.small_forgiveness_mul)
                if val_fall_2:
                    rise_increment = 0

            self.rise_counter += rise_increment
            self.both_counter += max(rise_increment, still_increment, 0.75)
        else:
            self.recalculate_delta_val(fall=val_fall)
            if val_still:
                self.forgive_rise(self.small_forgiveness_mul)
                still_increment = 1
                if val_decrease:
                    still_increment *= (1.0 - 0.15)
                self.still_counter += still_increment
                self.both_counter += still_increment
            else:
                self.update_best_val(val_loss)
                self.forgive_both(
                    min_forgiveness=self.forgive_rise() + self.forgive_still()
                )

        if val_rise or val_still:
            if val_fall_1 or train_fall:
                max(self.forgive_still(
                    self.mini_forgiveness_mul
                ), self.forgive_rise(
                    self.mini_forgiveness_mul
                ))
            elif val_fall_2:
                self.forgive_rise(self.mini_forgiveness_mul)
            if val_fall_1:
                self.update_best_val_2(val_loss)

        if self.rise_counter >= self.rise_patience:
            self.early_stop("rise", epoch)
        if self.still_counter >= self.still_patience:
            self.early_stop("still", epoch)
        if self.both_counter >= self.both_patience:
            self.early_stop("not falling", epoch)

        self.rise_counter = max(0, min(self.rise_patience, self.rise_counter))
        self.still_counter = max(0, min(self.still_patience, self.still_counter))
        self.both_counter = max(0, min(self.both_patience, self.both_counter))
        still_percent = self.still_counter / self.still_patience
        rise_percent = self.rise_counter / self.rise_patience
        both_percent = self.both_counter / self.both_patience

        if self.log_dir:
            self.still_writer.add_scalar(self.label + "patience", still_percent, global_step=epoch)
            self.rise_writer.add_scalar(self.label + "patience", rise_percent, global_step=epoch)
            self.both_writer.add_scalar(self.label + "patience", both_percent, global_step=epoch)

            self.still_writer.flush()
            self.rise_writer.flush()
            self.both_writer.flush()

        self.train_loss = train_loss
        self.val_loss = val_loss

        self.increment_epoch(epoch)

        return self.stopped

    def increment_epoch(self, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        if self.max_epoch and epoch >= self.max_epoch:
            self.stop()
            self.stop_reason = "max epoch"
            if self.debug >= 1:
                loss = f"best_val_loss_2={self.best_val_loss_2}" if self.update_state_mode == 2 else f"best_val_loss={self.best_val_loss}"
                print(f"INFO: Stopping at max epoch {epoch} with {loss} at epoch {self.best_epoch}")

        self.epoch = epoch + 1
        return self.epoch

    def recalculate_delta_val(self, fall=False):
        self.mid_val_loss, self.min_delta_val = self.calculate_interval(val=True)
        if (self.best_val_loss_2 - self.best_val_loss) < -self.min_delta_val:
            self.update_best_val(self.best_val_loss_2)
            if not fall:
                self.forgive_both(
                    mul=self.decent_forgiveness_mul,
                    min_forgiveness=self.forgive_still(
                        self.decent_forgiveness_mul
                    ) + self.forgive_rise(
                        self.decent_forgiveness_mul
                    )
                )

    def recalculate_delta_train(self):
        self.mid_train_loss, self.min_delta_train = self.calculate_interval(val=False)

    def update_state(self):
        self.best_state = deepcopy(self.model.state_dict())
        self.best_epoch = self.epoch

    def load_best_state(self):
        self.model.load_state_dict(deepcopy(self.best_state))

    def update_best_val_2(self, val_loss):
        self.best_val_loss_2 = val_loss
        if self.update_state_mode == 2:
            self.update_state()

    def update_best_train(self, train_loss):
        self.best_train_loss = train_loss

    def update_best_val(self, val_loss):
        self.best_val_loss = val_loss
        if val_loss < self.best_val_loss_2:
            self.update_best_val_2(val_loss)
        if self.update_state_mode == 1:
            self.update_state()

    def stop(self):
        if not self.stopped:
            self.model.load_state_dict(self.best_state)
            self.stopped = True
            self.last_epoch = self.epoch

    def early_stop(self, reason="idk", epoch=None):
        if not self.active:
            self.forgive_wait()
            return
        epoch = epoch if epoch is not None else self.epoch
        self.stop()
        self.stop_reason = reason
        loss = f"best_val_loss_2={self.best_val_loss_2}" if self.update_state_mode == 2 else f"best_val_loss={self.best_val_loss}"
        log = f"INFO: Early stopping due to {reason} at epoch {epoch} with {loss} at epoch {self.best_epoch}"
        if self.debug >= 1:
            if self.raise_ex:
                raise optuna.TrialPruned(log)
            else:
                print(log)

    def calculate_forgiveness(self, counter, forgiveness, patience):
        return min(counter, forgiveness * patience)

    def forgive_rise(self, mul=1):
        forgiveness = self.calculate_forgiveness(self.rise_counter, mul * self.rise_forgiveness, self.rise_counter)
        self.rise_counter -= forgiveness
        return forgiveness

    def forgive_still(self, mul=1):
        forgiveness = self.calculate_forgiveness(self.still_counter, mul * self.still_forgiveness, self.still_counter)
        self.still_counter -= forgiveness
        return forgiveness

    def forgive_both(self, mul=1, min_forgiveness=0):
        forgiveness = self.calculate_forgiveness(self.both_counter, mul * self.both_forgiveness, self.both_counter)
        forgiveness = min(self.both_counter, max(forgiveness, min_forgiveness))
        self.both_counter -= forgiveness
        return forgiveness

    def forgive_wait(self):
        if self.debug >= 2:
            print(f"INFO: Early stopping forgiven due to wait")

    def step_nan(self, epoch=None):
        if self.nan_counter < self.max_nan:
            self.nan_counter += 1
            if self.wait_counter < self.wait:
                self.wait_counter += 1
            self.increment_epoch(epoch)
            return True
        else:
            self.increment_epoch(epoch)
            return False
