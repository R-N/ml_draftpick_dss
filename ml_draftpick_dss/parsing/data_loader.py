
import os
import cv2
from .util import create_label_map
from .preprocessing import bgr2rgb, remove_artifact, circle_mask, BACKGROUNDS, rgba2rgb


def get_data(data_dir, img_size, labels, label_map=None, flip=False, artifact=False, circle=False, batch_size=128, backgrounds=BACKGROUNDS):
    data = []
    label_map = label_map or create_label_map(labels)
    for label in labels: 
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            try:
                if img.endswith(".db"):
                    continue
                img = os.path.join(path, img)
                if not os.path.isfile(img):
                    continue
                img = bgr2rgb(cv2.imread(img)) #convert BGR to RGB format
                img = cv2.resize(img, img_size) # Reshaping images to preferred size

                imgs = [img]

                if circle:
                    imgs = [circle_mask(img, color=bg) for bg in backgrounds]

                if flip:
                    imgs = [(img, cv2.flip(img, 1)) for img in imgs]
                    imgs = [j for i in imgs for j in i]

                if artifact:
                    imgs = [(
                        img,
                        remove_artifact(img, color=bg),
                        remove_artifact(img, invert=True, color=bg)
                    ) for bg in backgrounds for img in imgs]
                    imgs = [j for i in imgs for j in i]

                channels = imgs[0].shape[-1]
                if channels == 4:
                    imgs = [rgba2rgb(img) for img in imgs]
                
                #imgs = [tf.image.convert_image_dtype(img, tf.float32) for img in imgs]
                for img in imgs:
                    data.append([img, label_map[label]])

            except Exception as e:
                print(e)

    batch_size = batch_size or len(data)
    data = max(1, (batch_size//len(data))) * data
    return data#np.array(data)
