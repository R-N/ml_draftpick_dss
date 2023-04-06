import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.file_writers = {
            "train": tf.summary.create_file_writer(self.log_dir + f"/train"),
            "val": tf.summary.create_file_writer(self.log_dir + f"/val"),
        }
        
    def prepare_logging(self, log_dir="logs"):
        self.log_dir = log_dir

    def log_scalar(self, tag, value, step):
        tag_0 = tag[4:] if tag.startswith("val_") else tag
        writer = self.file_writers["val" if tag.startswith("val_") else "train"]
        with writer.as_default():
            tf.summary.scalar(tag_0, value, step=step)
            writer.flush()