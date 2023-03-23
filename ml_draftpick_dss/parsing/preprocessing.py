import cv2
import numpy as np

BACKGROUNDS = [
    (43, 61, 89),
    (57, 132, 182),
    #(59, 124, 169),
    #(48, 67, 90)
]
BORDERS = [
    (84, 161, 187),
    #(32, 120, 156)
]
TRANSLATIONS = [
    #(0, 0),
    (3, 0),
    (-3, 0),
    #(0, 3),
    #(0, -3),
    #(3, 3),
    #(3, -3),
    #(-3, 3),
    #(-3, -3)
]

def bgr2bgra(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

def circle_mask(img, color=BACKGROUNDS[0], alpha=False):
    img = img.copy()
    center = min(img.shape[:2]) // 2 
    radius = center
    img = bgr2bgra(img)

    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (center, center), radius, (255,255,255,255), -1)

    #show_imgs([mask])

    # apply mask to image
    #img[:, :, 3] = mask[:,:,3]
    img = cv2.bitwise_and(img, mask)
    img = img + cv2.bitwise_and((255 - mask), (*color, 255))
    if not alpha:
        img = rgba2rgb(img)
    return img

def circle_border(img, color=BORDERS[0], thickness=2):
    img = img.copy()
    center = min(img.shape[:2]) // 2 
    radius = center - (thickness//2)

    img = cv2.circle(img, (center, center), radius, color, thickness)

    return img

def translate(img, delta, background=BACKGROUNDS[0]):
    img = img.copy()
    translation_matrix = np.float32([ [1,0, delta[0]], [0,1,delta[1]] ])
    img = cv2.warpAffine(img, translation_matrix, img.shape[:2], borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    return img

def invert_x(tup, w):
    return (w-tup[0], *tup[1:])

CIRCLE_POS = (8, 16)
CIRCLE_RADIUS = 18
RECT_START = (8, 76)
RECT_END = (32, 100)

def remove_artifact(
    img, invert=False, scaler=None, 
    circle_pos=CIRCLE_POS, circle_radius=CIRCLE_RADIUS, 
    rect_start=RECT_START, rect_end=RECT_END,
    color=BACKGROUNDS[0], alpha=False
):
    img = img.copy()
    w = img.shape[1]
    if scaler:
        circle_pos = scaler.scale_point(circle_pos)
        circle_radius = scaler.scale_single(circle_radius)
        rect_start = scaler.scale_point(rect_start)
        rect_end = scaler.scale_point(rect_end)
    
    if invert:
        circle_pos = invert_x(circle_pos, w)
    img = cv2.circle(img, circle_pos, circle_radius, (*color, 255), -1)
    if invert:
        rect_start = invert_x(rect_start, w)
        rect_end = invert_x(rect_end, w)
    img = cv2.rectangle(img, rect_start, rect_end, (*color, 255), -1)
    if not alpha:
        img = rgba2rgb(img)
    return img

def rgba2rgb(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

def bgr2rgb(img):
    return img[...,::-1]

def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #img = cv2.filter2D(img, -1, kernel)
    #img = cv2.medianBlur(img, 5)
    img = cv2.filter2D(img, -1, kernel)
    return img

def upscale(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", 2)
    return sr.upsample(img)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def resize(img, resize):
    return cv2.resize(img, resize, interpolation=cv2.INTER_LANCZOS4)

def load_img(img, bgr=True, resize=None, keep_aspect_ratio=True):
    if isinstance(img, str):
        img = cv2.imread(img)
    if resize:
        if keep_aspect_ratio:
            resize = (resize[0], int(img.shape[1] * resize[0] / img.shape[0]))
        img = resize(img, tuple(reversed(resize)))
    if bgr:
        img = bgr2rgb(img)
    return img