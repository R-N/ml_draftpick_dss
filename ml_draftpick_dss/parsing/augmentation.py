import albumentations as A
import numpy as np
from functools import partial
import tensorflow as tf
from .util import loop_every_n
AUTOTUNE = tf.data.experimental.AUTOTUNE

TRANSFORMS = A.Compose([
    A.OneOf([
        A.GaussNoise(var_limit=(10,200)),
        A.ISONoise(),
        A.MultiplicativeNoise(),
    ], p=0.25),
    A.OneOf([
        A.AdvancedBlur(),
        A.Blur(),
        A.Defocus(radius=(3,5)),
        A.GaussianBlur(),
        A.MedianBlur(),
        A.GlassBlur(max_delta=2),
        A.MotionBlur(),
        A.RingingOvershoot(),
        A.ZoomBlur(),
        A.ElasticTransform(alpha=1.0, sigma=0.0, alpha_affine=0.0, border_mode=0, value=(255,255,255)),
    ], p=0.25),
    A.OneOf([
        A.Sharpen(),
        A.UnsharpMask (),
    ], p=0.25),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
    ], p=0.25),
    A.OneOf([
        A.CLAHE(),
        A.Emboss(),
        A.FancyPCA(),
        A.RGBShift(),
        A.RandomToneCurve(),
        A.HueSaturationValue(hue_shift_limit=(-5,5)),
    ], p=0.25),
    A.OneOf([
        A.ImageCompression(quality_lower=1),
        A.RingingOvershoot(blur_limit=(49, 99), cutoff=(2.0, 3.14)),
    ], p=0.25),
    A.OneOf([
        A.GridDistortion(border_mode=0, value=(255,255,255)),
        A.OpticalDistortion(border_mode=0, value=(255,255,255)),
        A.Perspective(),
        A.ElasticTransform(alpha_affine=18.0, border_mode=0, value=(255,255,255)),
        A.ShiftScaleRotate(rotate_limit=(0,0), border_mode=0, value=(255,255,255)),
    ], p=0.25),
])

def aug_fn(image, img_size):
    data = {"image":image}
    aug_data = TRANSFORMS(**data)
    aug_img = aug_data["image"]
    #aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.cast(aug_img, tf.float32)
    aug_img = tf.image.resize(aug_img, size=img_size)
    return aug_img

def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label

def set_shapes(img, label, img_size, label_count):
    img.set_shape((*img_size, 3))
    label.set_shape([label_count])
    return img, label

def separate_data(data):
    xs = [x for x, y in data]
    ys = [y for x, y in data]
    return xs, ys

def apply_aug(ds, img_size, label_count, shuffle_buffer=None, batch_size=32):
    shuffle_buffer = shuffle_buffer or len(ds)
    batch_size = batch_size or len(ds)
    ds = ds.map(partial(process_data, img_size=img_size), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    ds = ds.map(partial(set_shapes, img_size=img_size, label_count=label_count), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def prepare_data(data, img_size, label_count, shuffle_buffer=None, batch_size=32):
    shuffle_buffer = shuffle_buffer or len(data)
    batch_size = batch_size or len(data)
    xs, ys = separate_data(data)
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds = apply_aug(ds, img_size, label_count, batch_size=batch_size)
    return ds

def sample_augmented(data, n=1, i=0):
    batch = list(data.take(n))[i]
    augmented = batch[0]
    return augmented

def postprocess_augmented(augmented):
    return (augmented / (1 if np.max(augmented)<=1 else 255))

def loop_augmented(augmented, batch_size=5):
    return loop_every_n(augmented, batch_size)
