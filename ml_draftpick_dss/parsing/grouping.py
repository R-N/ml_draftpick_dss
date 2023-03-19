import os
from .preprocessing import sharpen, load_img
from .cropping import extract
from .ocr import OCR
from .scaler import Scaler
from .util import mkdir, inference_save_path, read_save_path, save_inference, exception_message, list_images
from .classifier import ScreenshotClassifier

BATCH_SIZE = 1+5

def create_batches(input_dir, *args, **kwargs):
    ss_list = list_images(input_dir)
    return _create_batches(input_dir, ss_list, *args, **kwargs)

def _create_batches(input_dir, ss_list, classifier=None, scaler=None, batch_size=BATCH_SIZE):
    n = len(ss_list)
    n = n // batch_size + (1 if n % batch_size > 0 else 0)

    if not classifier:
        ss_groups = [ss_list[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    else:
        ss_groups = []
        prev_batch = None
        for i in range(n):
            batch = ss_list[i*batch_size:i*batch_size+batch_size]
            ss_type, ss_type_img = infer_ss_type(os.path.join(input_dir, batch[0]), classifier, scaler)
            if ss_type != "History":
                if prev_batch:
                    index = find_history(input_dir, prev_batch[1:], classifier, scaler)
                    if index >= 0:
                        ss_groups[-1] = prev_batch[:1+index]
                        next_ss_list = ss_list[(i-1)*batch_size + 1+index:]
                        return ss_groups + _create_batches(input_dir, next_ss_list, classifier=classifier, scaler=scaler, batch_size=batch_size)
                index = find_history(input_dir, batch[1:], classifier, scaler)
                if index >= 0:
                    next_ss_list = ss_list[i*batch_size + 1+index:]
                    return ss_groups + _create_batches(input_dir, next_ss_list, classifier=classifier, scaler=scaler, batch_size=batch_size)
                prev_batch = None
                continue
            ss_groups.append(batch)
            prev_batch = batch
        if prev_batch:
            index = find_history(input_dir, prev_batch[1:], classifier, scaler)
            if index >= 0:
                ss_groups[-1] = prev_batch[:1+index]
                next_ss_list = ss_list[i*batch_size + 1+index:]
                return ss_groups + _create_batches(input_dir, next_ss_list, classifier=classifier, scaler=scaler, batch_size=batch_size)
    return ss_groups

def find_history(input_dir, history_candidates, classifier, scaler=None):
    candidate_ss_types = [infer_ss_type(os.path.join(input_dir, img), classifier, scaler) for img in history_candidates]
    candidate_ss_types = [t for t, i in candidate_ss_types]
    try:
        index = candidate_ss_types.index("History")
    except ValueError as ex:
        index = -1
    return index

def infer_ss_type(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    ss_corner_img = extract(img, "SS_CORNER", scaler=scaler, postprocessing=sharpen)
    ss_corner_class = classifier.infer([ss_corner_img])[0]
    return ss_corner_class, ss_corner_img

def read_player_name(img, ocr, scaler, bgr=True, throw=True):
    img = load_img(img, bgr=bgr)
    name_img = extract(img, "HISTORY_PLAYER_NAME", scaler=scaler, postprocessing=sharpen)
    try:
        name_text = ocr.read_history_player_name(name_img)
    except Exception as ex:
        message = exception_message(ex)
        if "Invalid SS" not in message or throw:
            raise
        name_text = None
    return name_text, name_img

def _generate_mv(ss_batch, input_dir, output_dir):
    return [(
        os.path.join(input_dir, s), 
        os.path.join(output_dir, s)
    ) for s in ss_batch]

def generate_mv(ss_batch, input_dir, output_dir, player_name=None, concat_input=False):
    if player_name:
        output_dir = os.path.join(output_dir, player_name)
        if concat_input:
            input_dir = os.path.join(input_dir, player_name)
    return _generate_mv(ss_batch, input_dir, output_dir)

class Grouper:
    def __init__(self, input_dir, output_dir, classifier, ocr=None, scaler=None, img=None, batch_size=BATCH_SIZE, inference_save_dir="inferences"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        mkdir(self.output_dir)
        self.inference_save_dir = inference_save_dir
        mkdir(self.inference_save_dir)
        self.batch_size = batch_size
        assert isinstance(classifier, ScreenshotClassifier)
        self.classifier = classifier
        scaler = scaler or (Scaler(img) if img else None)
        self._scaler = scaler
        self.scaler = None
        self.ocr = ocr or OCR(has_number=False)

    def inference_save_path(self, feature, infered_class, relpath, index=0):
        return inference_save_path(self.inference_save_dir, feature, infered_class, relpath, index=index)

    def read_save_path(self, feature, read, relpath, index=0):
        return read_save_path(self.inference_save_dir, feature, read, relpath, index=index)
    
    def input_relpath(self, path):
        return os.path.relpath(path, self.input_dir)

    def output_dir_player(self, player_name):
        return os.path.join(self.output_dir, player_name)

    def create_batches(self):
        return create_batches(self.input_dir, classifier=self.classifier, scaler=self.scaler, batch_size=self.batch_size)

    def infer_ss_type(self, img, bgr=True):
        return infer_ss_type(img, self.classifier, self.scaler, bgr=bgr)

    def read_player_name(self, img, bgr=True, throw=True):
        return read_player_name(img, self.ocr, self.scaler, bgr=bgr, throw=throw)
    
    def generate_mv(self, ss_batch, throw=True):
        img = os.path.join(self.input_dir, ss_batch[0])
        obj = self.infer(img, throw=throw)
        return self._generate_mv(ss_batch, obj, throw=throw)
    
    def _generate_mv(self, ss_batch, obj, throw=True):
        assert ((not throw) or (obj["ss_type"] == "History"))
        player_name = obj["player_name"]
        player_output_dir = self.output_dir_player(player_name)
        mvs = generate_mv(
            ss_batch,
            self.input_dir,
            player_output_dir
        )
        return player_output_dir, mvs
    
    def generate_groups(self, throw=True):
        batches = self.create_batches()
        for i, batch in enumerate(batches):
            yield (self.generate_mv(batch, throw=throw))

    def infer(self, img, bgr=True, throw=False, return_img=False):
        relpath = self.input_relpath(img)
        img = load_img(img, bgr=bgr)
        self.scaler = self._scaler or Scaler(img)
        ss_type, ss_type_img = self.infer_ss_type(img, bgr=False)
        player_name, player_name_img = self.read_player_name(img, bgr=False, throw=throw)
            
        obj = {
            "file": relpath,
            "ss_type": ss_type,
            "player_name": player_name,
        }
        if return_img:
            obj = {
                **obj,
                "ss_type_img": ss_type_img,
                "player_name_img": player_name_img
            }
        return obj
    
    def infer_all(self, throw=False, return_img=False):
        for img in list_images(self.input_dir):
            img = os.path.join(self.input_dir, img)
            obj = self.infer(img, throw=throw, return_img=return_img)
            yield obj

    def save_inference(self, obj):
        for feature in ["ss_type"]:
            save_inference(obj, self.inference_save_path, feature)
        for feature in ["player_name"]:
            save_inference(obj, self.read_save_path, feature)

class Grouper2(Grouper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_obj_batches(self):
        ss_list = list_images(self.input_dir)
        return self._create_obj_batches(ss_list)

    def _create_obj_batches(self, ss_list):
        objs = [self.infer(os.path.join(self.input_dir, img), throw=False) for img in ss_list]
        history_indexes = [i for i, obj in enumerate(objs) if obj["ss_type"] == "History"]
        indexes_2 = history_indexes + [None]
        index_pairs = [(indexes_2[i], indexes_2[i+1]) for i in range(len(history_indexes))]
        indexing = [(a, min(a+self.batch_size, b)) if b else (a, len(objs)) for a, b in index_pairs]
        obj_batches = [objs[a:b] for a, b in indexing]
        return obj_batches

    def create_batches(self):
        obj_batches = self.create_obj_batches()
        batches = [[obj["file"] for obj in objs] for objs in obj_batches]
        return batches
    
    def generate_groups(self, throw=True):
        obj_batches = self.create_obj_batches()
        for i, obj_batch in enumerate(obj_batches):
            batch = [obj["file"] for obj in obj_batch]
            yield (self._generate_mv(batch, obj_batch[0], throw=throw))
