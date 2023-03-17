import tensorflow as tf

import numpy as np

from .data_loader import get_data
from .util import create_label_map
from .augmentation import prepare_data
import tensorflow_addons as tfa
import json

METRICS = ["loss", "accuracy", "f1_score", "auc"]

def create_head_model_v2(label_count):
    return tf.keras.Sequential([
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(label_count, activation="softmax")      
    ], name="head")

class BaseClassifier:
    def __init__(self, labels, img_size, log_dir="logs", checkpoint_dir="checkpoints", metrics=METRICS):
        self.data_train = None
        self.data_val = None

        self.labels = labels
        self.label_count = len(labels)
        self.label_map = create_label_map(labels)

        assert img_size[0]%32 == 0 and img_size[1]%32 == 0
        self.img_size = img_size

        self.metrics = metrics + [f"val_{m}" for m in metrics]

        self.base_model = None
        self.head_model = None
        self.model = None
        self.optim = None
        self.compiled = False
        self.create_model()

        self.checkpoint = None
        self.checkpoint_manager = None
        self.prepare_checkpoint()

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.file_writers = None

        self.best_metrics = {m: 100 if "loss" in m else 0 for m in self.metrics}
        self.best_metrics["epoch"] = 0
        

    def load_data(self, train_dir, val_dir=None, augment_val=True, flip=False, artifact=False, circle=False, train_batch_size=128, val_batch_size=None):
        val_dir = val_dir or train_dir
        val_batch_size = val_batch_size or train_batch_size

        data_train = get_data(train_dir, flip=flip, artifact=artifact, circle=circle, batch_size=train_batch_size)
        data_val = get_data(val_dir, flip=flip, artifact=artifact, circle=circle, batch_size=val_batch_size)
        
        self.data_train = prepare_data(data_train, self.img_size, batch_size=train_batch_size)
        if augment_val:
            self.data_val = prepare_data(data_val, self.img_size, batch_size=val_batch_size)
        else:
            raise Exception("Not implemented")
        
    @property
    def input_shape(self):
        return (*self.img_size, 3)

    def _create_base_model(self):
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape = self.input_shape, 
            include_top = False, 
            weights = "imagenet",
            include_preprocessing=True
        )
        base_model.trainable = False
        return base_model
    
    def _create_head_model(self):
        head = create_head_model_v2(self.label_count)
        return head
    
    def create_model(self):
        self.base_model = self._create_base_model()
        self.head_model = self._create_head_model()

        # Create new model on top
        input = tf.keras.layers.Input(shape=self.input_shape)
        x = input
        #x = scale_layer(x)
        x = self.base_model(x, training=False)
        x = self.head_model(x)
        output = x
        #output = head(x)
        model = tf.keras.Model(input, output)

        self.model = model
        self.compiled = False

        return model
    
    def compile(self, base_lr=1e-3):
        assert self.model

        f1 = tfa.metrics.F1Score(num_classes=self.label_count, average=None)

        self.optim = tf.keras.optimizers.Adam(learning_rate=base_lr)
        self.model.compile(
            optimizer=self.optim,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', f1, "AUC"]
        )

        self.compiled = True

    def prepare_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.head_model, optimizer=self.optim)
        self.checkpoint_managers = {m: tf.train.CheckpointManager(self.checkpoint, f"{self.checkpoint_dir}/{m}", max_to_keep=1) for m in self.metrics}

    def prepare_logging(self):
        self.file_writers = {
            "train": tf.summary.create_file_writer(self.log_dir + f"/train"),
            "val": tf.summary.create_file_writer(self.log_dir + f"/val"),
        }


    def log_scalar(self, tag, value, step):
        tag_0 = tag[4:] if tag.startswith("val_") else tag
        writer = self.file_writers["val" if tag.startswith("val_") else "train"]
        with writer.as_default():
            tf.summary.scalar(tag_0, value, step=step)
            writer.flush()

    def load_checkpoint(self, checkpoint="loss"):
        assert self.model
        self.checkpoint_managers[checkpoint].restore_or_initialize()
        self.load_best_metrics(f"checkpoints/{checkpoint}/metrics.json")

    def load_best_metrics(self, path):
        try:
            with open(path, 'r') as f:
                self.best_metrics = json.load(f)
        except Exception as ex:
            print(ex)
    
    def prepare_training(self, load_checkpoint="loss"):
        assert self.checkpoint_manager
        assert self.compiled
        assert self.file_writers

        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
            
    def set_lr(self, lr):
        self.optim.lr.assign(lr)

    @property
    def epoch(self):
        return self.best_metrics["epoch"]
    
    def save_best_metrics(self, path):
        with open(path, 'w') as f:
            json.dump(self.best_metrics, f)

    def train_epoch(self, callbacks=[]):
        epoch = self.epoch

        history = self.model.fit(
            self.data_train, 
            epochs=1 , 
            validation_data=self.data_val, 
            callbacks=callbacks
        )
        cur_metrics = {m: history.history[m][0] for m in self.metrics}
        cur_metrics = {m: float(v if tf.rank(v) == 0 else tf.math.reduce_mean(v)) for m, v in cur_metrics.items()}

        for m, v in cur_metrics.items():
            self.log_scalar(m, v, epoch)

        new_best_metrics = []
        for m in self.metrics:
            cur_val, best_val = cur_metrics[m], self.best_metrics[m]
            if "loss" in m:
                cur_val, best_val = -cur_val, -best_val
            if cur_val > best_val:
                self.best_metrics[m] = cur_metrics[m]
                new_best_metrics.append((m, self.best_metrics[m], cur_metrics[m]))

        self.save_best_metrics("metrics.json")

        for m, old, new in new_best_metrics:
            self.checkpoint_managers[m].save()
            self.save_best_metrics(f"checkpoints/{m}/metrics.json")

        self.best_metrics["epoch"] += 1

        return epoch, history, new_best_metrics

    def save(self, *args, **kwagrs):
        self.model.save(*args, **kwargs)

    def prepare_imgs(self, imgs):
        assert imgs.shape[-1] == 3
        imgs = [tf.cast(img, tf.float32) for img in imgs]
        imgs = [tf.image.resize(img, size=self.img_size) for img in imgs]
        imgs = tf.stack(imgs)
        return imgs

    def infer(self, imgs):
        imgs = self.prepare_imgs(imgs)
        return [self.labels[np.argmax(y)] for y in self.model(imgs)]

MATCH_RESULT_LABELS = ["Victory", "Defeat", "Invalid"]
MATCH_RESULT_LIST_IMG_SIZE = (96, 224)

class MatchResultListClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MATCH_RESULT_LABELS, 
            MATCH_RESULT_LIST_IMG_SIZE, 
            *args,
            **kwargs
        )
        
MATCH_RESULT_IMG_SIZE = (96, 224)

class MatchResultClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MATCH_RESULT_LABELS, 
            MATCH_RESULT_IMG_SIZE, 
            *args,
            **kwargs
        )

MEDAL_LABELS = ["MVP Win", "MVP Lose", "Gold", "Silver", "Bronze", "AFK"]
MEDAL_IMG_SIZE = (160, 224)

class MedalClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MEDAL_LABELS, 
            MEDAL_IMG_SIZE, 
            *args,
            **kwargs
        )

HERO_ICON_LABELS = ['Miya', 'Balmond', 'Saber', 'Alice', 'Nana', 'Tigreal', 'Alucard', 'Karina', 'Akai', 'Franco', 'Bane', 'Bruno', 'Clint', 'Rafaela', 'Eudora', 'Zilong', 'Fanny', 'Layla', 'Minotaur', 'Lolita', 'Hayabusa', 'Freya', 'Gord', 'Natalia', 'Kagura', 'Chou', 'Sun', 'Alpha', 'Ruby', 'Yi Sun-shin', 'Moskov', 'Johnson', 'Cyclops', 'Estes', 'Hilda', 'Aurora', 'Lapu-Lapu', 'Vexana', 'Roger', 'Karrie', 'Gatotkaca', 'Harley', 'Irithel', 'Grock', 'Argus', 'Odette', 'Lancelot', 'Diggie', 'Hylos', 'Zhask', 'Helcurt', 'Pharsa', 'Lesley', 'Jawhead', 'Angela', 'Gusion', 'Valir', 'Martis', 'Uranus', 'Hanabi', "Chang'e", 'Kaja', 'Selena', 'Aldous', 'Claude', 'Vale', 'Leomord', 'Lunox', 'Hanzo', 'Belerick', 'Kimmy', 'Thamuz', 'Harith', 'Minsitthar', 'Kadita', 'Faramis', 'Badang', 'Khufra', 'Granger', 'Guinevere', 'Esmeralda', 'Terizla', 'X.Borg', 'Ling', 'Dyrroth', 'Lylia', 'Baxia', 'Masha', 'Wanwan', 'Silvanna', 'Carmilla', 'Cecilion', 'Atlas', 'Popol and Kupa', 'Yu Zhong', 'Luo Yi', 'Benedetta', 'Khaleed', 'Barats', 'Brody', 'Yve', 'Mathilda', 'Paquito', 'Gloo', 'Beatrix', 'Phoveus', 'Natan', 'Aulus', 'Aamon', 'Floryn', 'Valentina', 'Edith', 'Yin', 'Melissa', 'Xavier', 'Julian', 'Fredrinn', 'Joy', 'Novaria', 'Arlott']
HERO_ICON_IMG_SIZE = (96, 96)

class HeroIconClassifier(BaseClassifier):
    def __init__(self, *args, labels=HERO_ICON_LABELS, **kwargs):
        super().__init__(
            labels, 
            HERO_ICON_IMG_SIZE, 
            *args,
            **kwargs
        )