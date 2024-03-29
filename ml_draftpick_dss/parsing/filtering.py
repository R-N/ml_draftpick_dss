import os
from .preprocessing import sharpen, load_img
from .cropping import extract
from .ocr import OCR, DEFAULT_SIMILARITY
from .scaler import Scaler
from .grouping import BATCH_SIZE, create_batches, generate_mv as _generate_mv, infer_ss_type, Grouper2, read_opening_failure, check_opening_failure
from .classifier import MatchResultListClassifier, ScreenshotClassifier
from .util import inference_save_path, read_save_path, save_inference, mkdir, list_subdirectories, exception_message

def read_match_types(img, ocr, scaler, batch_index=0, bgr=True):
    img = load_img(img, bgr=bgr)
    match_type_imgs = extract(img, "MATCH_TYPE_LIST", scaler=scaler, batch_index=batch_index%4, split_list=True, crop_list=True)
    #match_type_texts = [ocr.read(i) for i in match_type_imgs]
    match_type_texts = ocr.read(match_type_imgs)
    return match_type_texts, match_type_imgs

def infer_match_results(img, classifier, scaler, batch_index=0, bgr=True):
    img = load_img(img, bgr=bgr)
    match_result_imgs = extract(img, "MATCH_RESULT_LIST", scaler=scaler, batch_index=batch_index%4, split_list=True, crop_list=True)
    match_result_classes = classifier.infer(match_result_imgs)
    return match_result_classes, match_result_imgs

def generate_mask(match_types, match_results, similarity=DEFAULT_SIMILARITY):
    assert len(match_types) == len(match_results)
    match_results_mask = [c in {"Victory", "Defeat"} for c in match_results]
    match_types_mask = [similarity(s, "ranked") >= 0.8 for s in match_types]
    final_mask = [(match_results_mask[i] and match_types_mask[i]) for i in range(len(match_types))]
    return final_mask

def filter_batch(ss_batch, mask):
    return [ss_batch[i+1] for i in range(len(ss_batch)-1) if mask[i]]

def generate_cp(ss_batch, input_dir, output_dir, player_name):
    return _generate_mv(ss_batch, input_dir, output_dir, player_name, concat_input=True)

class Filterer:
    def __init__(self, input_dir, output_dir, ss_classifier, result_list_classifier, ocr=None, img_size=None, batch_size=BATCH_SIZE, inference_save_dir="inferences"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        mkdir(self.output_dir)
        self.inference_save_dir = inference_save_dir
        mkdir(self.inference_save_dir)
        self.batch_size = batch_size
        assert isinstance(ss_classifier, ScreenshotClassifier)
        self.ss_classifier = ss_classifier
        assert isinstance(result_list_classifier, MatchResultListClassifier)
        self.result_list_classifier = result_list_classifier
        self.img_size = img_size
        scaler = Scaler(img_size) if img_size else None
        self._scaler = scaler
        self.scaler = None
        self.ocr = ocr or OCR(has_number=False)

    def inference_save_path(self, feature, infered_class, relpath, index=0):
        return inference_save_path(self.inference_save_dir, feature, infered_class, relpath, index=index)

    def read_save_path(self, feature, read, relpath, index=0):
        return read_save_path(self.inference_save_dir, feature, read, relpath, index=index)

    def output_dir_player(self, player_name):
        return os.path.join(self.output_dir, player_name)

    def input_dir_player(self, player_name):
        return os.path.join(self.input_dir, player_name)

    def create_batches(self, player_name):
        return create_batches(self.input_dir_player(player_name), self.ss_classifier, self.ocr, self.scaler, batch_size=self.batch_size)

    def infer_ss_type(self, img, bgr=True):
        return infer_ss_type(img, self.ss_classifier, self.scaler, bgr=bgr)

    def read_match_types(self, img, batch_index=0, bgr=True):
        return read_match_types(img, self.ocr, self.scaler, batch_index=batch_index%4, bgr=bgr)

    def infer_match_results(self, img, batch_index=0, bgr=True):
        return infer_match_results(img, self.result_list_classifier, self.scaler, batch_index=batch_index%4, bgr=bgr)

    def read_opening_failure(self, img, bgr=True):
        return read_opening_failure(img, self.ocr, scaler=self.scaler, bgr=bgr)

    def check_opening_failure(self, text, similarity=DEFAULT_SIMILARITY):
        return check_opening_failure(text, similarity=similarity)

    def generate_cp(self, ss_batch, player_name, batch_index=0, throw=False):
        img = os.path.join(self.input_dir_player(player_name), ss_batch[0])
        obj = self.infer(img, batch_index=batch_index, throw=throw)
        mask = generate_mask(obj["match_types"], obj["match_results"])
        valid_ss = filter_batch(ss_batch, mask)
        player_input_dir = self.input_dir_player(player_name)
        player_output_dir = self.output_dir_player(player_name)
        cps = _generate_mv(
            valid_ss,
            player_input_dir,
            player_output_dir
        )
        return cps
    
    def _generate_cp_player(self, player_name):
        batches = self.create_batches(player_name)
        player_output_dir = self.output_dir_player(player_name)
        cps = [self.generate_cp(batch, player_name, i%4) for i, batch in enumerate(batches)]
        cps = [j for i in cps for j in i]
        return player_output_dir, cps
    
    def _generate_cp_player_split(self, player_name):
        batches = self.create_batches(player_name)
        player_output_dir = self.output_dir_player(player_name)
        cps, result_list, opening_failure_list = [], [], []
        for i, batch in enumerate(batches):
            try:
                cp = self.generate_cp(batch, player_name, i%4)
                cps.append(cp)
            except AssertionError as ex:
                message = exception_message(ex)
                relpath = self.input_relpath(os.path.join(player_output_dir, batch[0]))
                if message.startswith("RESULT"):
                    print(message)
                    result_list.append(relpath)
                if message.startswith("OPENING_FAILURE"):
                    print(message)
                    opening_failure_list.append(relpath)
                else:
                    raise
        cps = [j for i in cps for j in i]
        return player_output_dir, cps, result_list, opening_failure_list
    
    def generate_cp_player(self, player_name, split=True, throw=False):
        if split:
            return self._generate_cp_player_split(player_name)
        return self._generate_cp_player(player_name, throw=throw)

    def generate_cp_all(self, split=True, throw=False, start=None, exclude={}):
        players = list_subdirectories(self.input_dir)
        if start:
            players = players[players.index(start):]
        if exclude:
            players = [p for p in players if p not in exclude]
        for player in players:
            print("Generating", player)
            yield self.generate_cp_player(player, split=split, throw=throw)
    
    def input_relpath(self, path):
        return os.path.relpath(path, self.input_dir)
    
    def infer(self, img, batch_index=0, bgr=True, throw=False, return_img=False):
        relpath = self.input_relpath(img)
        img = load_img(img, bgr=bgr, resize=self.img_size)
        self.scaler = self._scaler or Scaler(img)

        ss_type, ss_type_img = self.infer_ss_type(img, bgr=False)
        assert ((not throw) or (ss_type=="History")), f"RESULT: {relpath}"

        opening_failure_text, opening_failure_img = self.read_opening_failure(img, bgr=False)
        opening_failure = self.check_opening_failure(opening_failure_text)
        assert ((not throw) or (not opening_failure)), f"OPENING_FAILURE: {relpath}"

        match_types, match_types_img = self.read_match_types(img, batch_index=batch_index%4, bgr=False)
        match_results, match_results_img = self.infer_match_results(img, batch_index=batch_index%4, bgr=False)
        obj = {
            "file": relpath,
            "match_types": match_types,
            "match_results": match_results,
            "ss_type": ss_type,
            "opening_failure": opening_failure
        }
        if return_img:
            obj = {
                **obj,
                "match_types_img": match_types_img,
                "match_results_img": match_results_img,
                "ss_type_img": ss_type_img,
                "opening_failure_img": opening_failure_img
            }
        return obj
    
    def infer_player(self, player_name, throw=False, return_img=False):
        batches = self.create_batches(player_name)
        imgs = [os.path.join(self.input_dir_player(player_name), ss_batch[0]) for ss_batch in batches]
        objs = [self.infer(img, i%4, throw=throw, return_img=return_img) for i, img in enumerate(imgs)]
        return objs

    def infer_all(self, throw=False, return_img=False, start=None, exclude={}):
        players = list_subdirectories(self.input_dir)
        if start:
            players = players[players.index(start):]
        if exclude:
            players = [p for p in players if p not in exclude]
        for player in players:
            print("Infering", player)
            yield self.infer_player(player, throw=throw, return_img=return_img)

    def save_inference(self, obj):
        for feature in ["ss_type", "match_results"]:
            save_inference(obj, self.inference_save_path, feature)
        for feature in ["opening_failure", "match_types"]:
            save_inference(obj, self.read_save_path, feature)


class Filterer2(Filterer):
    def __init__(self, input_dir, output_dir, ss_classifier, *args, **kwargs):
        super().__init__(input_dir, output_dir, ss_classifier=ss_classifier, *args, **kwargs)
        self.grouper = Grouper2(f"{input_dir}_group", f"{output_dir}_group", classifier=self.ss_classifier, ocr=self.ocr, img_size=self.img_size, batch_size=self.batch_size)

    def create_batches(self, player_name):
        self.grouper.input_dir = self.input_dir_player(player_name)
        return self.grouper.create_batches()