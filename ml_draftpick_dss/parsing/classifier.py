import tensorflow as tf

import numpy as np

from .data_loader import get_data
from .util import create_label_map
from .augmentation import create_dataset, augment_dataset
import tensorflow_addons as tfa
import json
from ..constants import HERO_LIST
from .preprocessing import rgba2rgb, TRANSLATIONS, BORDERS

METRICS = ["loss", "accuracy", "f1_score", "auc"]

def create_base_model_v1(input_shape, label_count, include_top=False, trainable=False):
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape = input_shape, 
        include_top = include_top, 
        weights = "imagenet",
        classes=label_count,
        include_preprocessing=True
    )
    base_model.trainable = trainable
    return base_model

def create_base_model_v2(input_shape, label_count, include_top=False, trainable=False):
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape = input_shape, 
        include_top = include_top, 
        weights = "imagenet",
        classes=label_count,
        include_preprocessing=True
    )
    base_model.trainable = trainable
    return base_model

def create_head_model_v2(label_count):
    return tf.keras.Sequential([
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(label_count, activation="softmax")      
    ], name="head")

def create_head_model_v3(label_count):
    return tf.keras.Sequential([
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(label_count, activation="softmax")      
    ], name="head")

class BaseClassifier:
    def __init__(self, labels, img_size, log_dir="logs", checkpoint_dir="checkpoints", metrics=METRICS, base_model_factory=create_base_model_v1, head_model_factory=create_head_model_v2):
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
        self.base_model_factory = base_model_factory or create_base_model_v1
        self.head_model_factory = head_model_factory or create_head_model_v2
        self.create_model()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = None
        self.checkpoint_manager = None
        self.prepare_checkpoint()

        self.log_dir = log_dir
        self.file_writers = None

        self.best_metrics = {m: 100 if "loss" in m else 0 for m in self.metrics}
        self.best_metrics["epoch"] = 0
        

    def load_data(self, train_dir, val_dir=None, augment_val=True, flip=False, artifact=False, circle=False,  circle_border=False, translate=False, train_batch_size=32, val_batch_size=None, translations=TRANSLATIONS, borders=BORDERS, max_per_class=32):
        val_dir = val_dir or train_dir
        val_batch_size = val_batch_size or train_batch_size

        print("A")
        self.data_train = get_data(train_dir, self.img_size, self.labels, flip=flip, artifact=artifact, circle=circle, circle_border=circle_border, translate=translate, batch_size=train_batch_size, translations=translations, borders=borders, max_per_class=max_per_class)
        print("B")
        self.data_train = create_dataset(self.data_train)
        print("C")
        if train_dir == val_dir and train_batch_size == val_batch_size:
            self.data_val = self.data_train
        else:
            print("D")
            self.data_val = get_data(val_dir, self.img_size, self.labels, flip=flip and augment_val, artifact=artifact and augment_val, circle=circle and augment_val, circle_border=circle_border and augment_val, translate=translate and augment_val, batch_size=val_batch_size, translations=translations, borders=borders, max_per_class=max_per_class)
            print("E")
            self.data_val = create_dataset(self.data_val)
            print("F")

        print("G")
        self.data_train = augment_dataset(self.data_train, self.img_size, self.label_count, batch_size=train_batch_size)
        print("H")
        if augment_val:
            if train_dir == val_dir and train_batch_size == val_batch_size:
                self.data_val = self.data_train
            else:
                print("I")
                self.data_val = augment_dataset(self.data_val, self.img_size, self.label_count, batch_size=val_batch_size)
                print("J")
        
    @property
    def input_shape(self):
        return (*self.img_size, 3)
    
    def create_model(self):
        create_head_model = callable(self.head_model_factory)
        self.base_model = self.base_model_factory(self.input_shape, self.label_count, include_top=not create_head_model, trainable=not create_head_model)
        self.head_model = self.head_model_factory(self.label_count)

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
        assert self.model
        kwargs = {
            "optimizer": self.optim
        } if self.optim else {}
        self.checkpoint = tf.train.Checkpoint(model=self.head_model, **kwargs)
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
        assert self.checkpoint_managers
        res = self.checkpoint_managers[checkpoint].restore_or_initialize()
        print("restored checkpoint: ", res)
        self.load_best_metrics(f"{self.checkpoint_dir}/{checkpoint}/metrics.json")

    def save_checkpoint(self, checkpoint):
        assert self.checkpoint_managers
        self.checkpoint_managers[checkpoint].save()
        self.save_best_metrics(f"{self.checkpoint_dir}/{checkpoint}/metrics.json")

    def load_best_metrics(self, path):
        try:
            with open(path, 'r') as f:
                self.best_metrics = json.load(f)
        except Exception as ex:
            print(ex)
    
    def prepare_training(self, load_checkpoint="loss"):
        assert self.compiled
        assert self.checkpoint_managers
        assert self.file_writers

        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
            
    def set_lr(self, lr):
        return self.optim.lr.assign(lr)

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
                new_best_metrics.append((m, self.best_metrics[m], cur_metrics[m]))
                self.best_metrics[m] = cur_metrics[m]

        self.save_best_metrics("metrics.json")

        for m, old, new in new_best_metrics:
            self.save_checkpoint(m)

        self.best_metrics["epoch"] += 1

        return epoch, history, new_best_metrics

    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)

    def prepare_imgs(self, imgs):
        if imgs[0].shape[-1] == 4:
            imgs = [rgba2rgb(img) for img in imgs]
        imgs = [tf.cast(img, tf.float32) for img in imgs]
        imgs = [tf.image.resize(img, size=self.img_size) for img in imgs]
        imgs = tf.stack(imgs)
        return imgs

    def infer(self, imgs):
        imgs = self.prepare_imgs(imgs)
        return [self.labels[np.argmax(y)] for y in self.model(imgs)]

MATCH_RESULT_LIST_LABELS = ["Victory", "Defeat", "Invalid", "AFK", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "Bad"]
MATCH_RESULT_LIST_IMG_SIZE = (96, 224)

class MatchResultListClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MATCH_RESULT_LIST_LABELS, 
            MATCH_RESULT_LIST_IMG_SIZE, 
            *args,
            **kwargs
        )
        
MATCH_RESULT_LABELS = ["Victory", "Defeat", "Invalid", "Bad"]
MATCH_RESULT_IMG_SIZE = (96, 224)

class MatchResultClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MATCH_RESULT_LABELS, 
            MATCH_RESULT_IMG_SIZE, 
            *args,
            **kwargs
        )

MEDAL_LABELS = ["MVP Win", "MVP Lose", "Gold", "Silver", "Bronze", "AFK", "Bad"]
MEDAL_IMG_SIZE = (160, 224)

class MedalClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            MEDAL_LABELS, 
            MEDAL_IMG_SIZE, 
            *args,
            **kwargs
        )

SS_LABELS = ["History", "Result"]
SS_IMG_SIZE = (96, 224)

class ScreenshotClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(
            SS_LABELS, 
            SS_IMG_SIZE, 
            *args,
            **kwargs
        )

HERO_ICON_LABELS = HERO_LIST + ["Bad"]
HERO_ICON_IMG_SIZE = (96, 96)

class HeroIconClassifier(BaseClassifier):
    def __init__(self, *args, labels=HERO_ICON_LABELS, base_model_factory=create_base_model_v1, head_model_factory=create_head_model_v3, **kwargs):
        super().__init__(
            labels, 
            HERO_ICON_IMG_SIZE, 
            *args,
            base_model_factory=base_model_factory,
            head_model_factory=head_model_factory,
            **kwargs
        )

    def load_data(self, *args, flip=True, artifact=True, circle=True, circle_border=True, translate=True, train_batch_size=None, **kwargs):
        super().load_data(*args, flip=flip, artifact=artifact, circle=circle, circle_border=circle_border, translate=translate, train_batch_size=train_batch_size, **kwargs)