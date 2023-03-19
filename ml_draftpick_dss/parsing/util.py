import matplotlib.pyplot as plt
import numpy as np
from .preprocessing import load_img
import os
import cv2
from pathlib import Path

def show_imgs(imgs, cols=10, fig_title="", show=True):
    rows = (len(imgs) // cols) + (1 if len(imgs) % cols > 0 else 0)
    fig = plt.figure(figsize=(16, 3*rows))
    if fig_title:
        fig.suptitle(fig_title, fontsize=32)
    for i in range(0, len(imgs)):
        #img = get_nth(heroes, i)
        img = imgs[i]
        if isinstance(img, tuple):
            img = img[-1]

        ax = plt.subplot(rows, cols, i+1)

        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_axis_off()
    if cols > 5:
        fig.tight_layout()
    if show:
        plt.show()
    return fig

def show_image_file(f, show=True):
    img = load_img(f)
    return show_imgs([img], cols=1, height=10, show=show)


def create_label_map(labels):
    label_count = len(labels)
    return dict(zip(labels, [np.array([1 if i==j else 0 for j in range(label_count)]) for i in range(label_count)]))

def loop_every_n(arr, n):
    for i in range(len(arr)//n):
        yield arr[i*n:n+i*n]

def split_extension(path):
    return path.rsplit(".", maxsplit=1)

def inference_save_path(save_dir, feature, infered_class, relpath, index=0):
    file_name = split_extension(relpath.replace("/", "_"))
    file_name = f"{file_name[-2]}_{index}.{file_name[-1]}"
    return os.path.join(save_dir, feature, infered_class, file_name)

def read_save_path(save_dir, feature, read, relpath, index=0):
    file_name = split_extension(relpath.replace("/", "_"))
    file_name = f"{read}_{file_name[-2]}_{index}.{file_name[-1]}"
    return os.path.join(save_dir, feature, file_name)

def listify(x):
    return x if isinstance(x, list) else [x]

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def save_inference(obj, path_factory, feature):
    img_key = f"{feature}_img"
    pair = list(zip(listify(obj[feature]), listify(obj[img_key])))
    for i, (inference, img) in enumerate(pair):
        path = path_factory(feature, inference, obj["file"], index=i)
        dir = os.path.dirname(path)
        mkdir(dir)
        cv2.imwrite(path, rgb2bgr(img))

def mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def exception_message(ex):
    return ex.message if hasattr(ex, "message") else str(ex)

def list_subdirectories(dir):
    subd = os.listdir(dir)
    subd = [d for d in subd if os.path.isdir(os.path.join(dir, d))]
    return subd

def list_images(dir, extension=".jpg"):
    imgs = os.listdir(dir)
    imgs = [i for i in imgs if os.path.isfile(os.path.join(dir, i) and i.lower().endswith(extension))]
    return imgs