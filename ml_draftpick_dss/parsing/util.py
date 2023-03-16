import matplotlib.pyplot as plt
import numpy as np

def show_imgs(imgs, cols=10, fig_title=""):
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
    return fig

def show_image_file(f):
    img = cv2.imread(f)[...,::-1]
    show_imgs([img], cols=1, height=10)


def create_label_map(labels):
    label_count = len(labels)
    return dict(zip(labels, [np.array([1 if i==j else 0 for j in range(label_count)]) for i in range(label_count)]))

def loop_every_n(arr, n):
    for i in range(len(arr)//n):
        yield arr[i*n:n+i*n]
