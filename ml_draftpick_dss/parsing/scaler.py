REFERENCE_RESOLUTION = (1080, 1920)

def offset_offset(tup, x):
    x = int(x)
    tup = tuple(tup)
    return (tup[0] + x, tup[1] - x)

class Scaler:
    def __init__(self, img, reference_resolution=REFERENCE_RESOLUTION):
        if not isinstance(img, tuple) and not isinstance(img, list):
            img = img.shape[:2]
        self.img_size = img
        self.nonwide_ratio = self.result_size[0] / reference_resolution[0]
        extra_side = 0.5 * self.result_size[1] * ((self.result_size[1]/self.result_size[0]) / (reference_resolution[1]/reference_resolution[0]) - 1)
        extra_side = 0.5 * (self.result_size[1] - reference_resolution[1] * self.nonwide_ratio) 
        self.extra_side = int(extra_side)

    def scale(self, value, x=False, offset=0):
        if x:
            ret = tuple(int(self.extra_side + value[i] * self.nonwide_ratio) for i in range(len(value)))
            ret = offset_offset(ret, offset)
        else:
            ret = tuple(int(value[i] * self.nonwide_ratio) for i in range(len(value)))
        return ret