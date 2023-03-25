
import os
import cv2
import random
from .util import create_label_map
from .preprocessing import bgr2rgb, remove_artifact, circle_mask, BACKGROUNDS, rgba2rgb, circle_border as _circle_border, BORDERS, translate as _translate, TRANSLATIONS, load_img


def get_data(data_dir, img_size, labels, label_map=None, flip=False, artifact=False, circle=False, circle_border=False, translate=False, batch_size=128, backgrounds=BACKGROUNDS, borders=BORDERS, translations=TRANSLATIONS, max_per_class=32, whitelist={}):
    data = []
    label_map = label_map or create_label_map(labels)
    if whitelist:
        labels = [label for label in labels if label in whitelist]
        assert len(labels) == len(whitelist)
    for label in labels: 
        path = os.path.join(data_dir, label)
        files = os.listdir(path)
        files = [os.path.join(path, f) for f in files if not (f.endswith(".db") or f.endswith(".ini"))]
        files = [f for f in files if os.path.isfile(f)]
        truths = [f for f in files if "ground_truth" in f]
        files = [f for f in files if f not in truths]
        files = random.sample(files, min(len(files), max_per_class - len(truths)))
        files = files + truths
        assert len(files) <= max_per_class
        for img in files:
            try:
                img_0 = img
                img = load_img(img) #convert BGR to RGB format
                img = cv2.resize(img, img_size) # Reshaping images to preferred size

                imgs = [img]

                if circle_border:
                    imgs = imgs + [_circle_border(img, color=border) for border in borders for img in imgs]
                if flip:
                    imgs = imgs + [cv2.flip(img, 1) for img in imgs]

                if translate or circle or artifact:
                    imgs_0 = list(imgs)
                    imgs_n = list(imgs)
                    for bg in backgrounds:
                        imgs_i = imgs_0
                        if translate:
                            imgs_i = imgs_i + [_translate(img, delta, background=bg) for delta in translations for img in imgs_i]
                        if circle:
                            imgs_i = [circle_mask(img, color=bg) for img in imgs_i]
                        if artifact:
                            imgs_i = [(
                                img,
                                remove_artifact(img, color=bg),
                                remove_artifact(img, invert=True, color=bg),
                                remove_artifact(remove_artifact(img, color=bg), invert=True, color=bg)
                            ) for img in imgs_i]
                            imgs_i = [j for i in imgs_i for j in i]
                        imgs_n.extend(imgs_i)
                    imgs = imgs_n

                channels = imgs[0].shape[-1]
                if channels == 4:
                    imgs = [rgba2rgb(img) for img in imgs]
                
                #imgs = [tf.image.convert_image_dtype(img, tf.float32) for img in imgs]
                for img in imgs:
                    data.append([img, label_map[label]])
            except Exception as e:
                print(img_0)
                raise
                print(e)

    batch_size = batch_size or len(data)
    data = max(1, (batch_size//len(data))) * data
    random.shuffle(data)
    return data#np.array(data)
