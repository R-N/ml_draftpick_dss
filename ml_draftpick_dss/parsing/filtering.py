import os
from .preprocessing import sharpen, load_img
from .cropping import extract
from .ocr import OCR, DEFAULT_SIMILARITY
from .scaler import Scaler
from .grouping import BATCH_SIZE, create_batches, generate_mv as _generate_mv
from .classifier import MatchResultListClassifier

def read_match_types(img, ocr, scaler, batch_index=0, bgr=True):
    img = load_img(img, bgr=bgr)
    match_type_imgs = extract(img, "MATCH_TYPE_LIST", scaler=scaler, batch_index=batch_index%4, split_list=True, crop_list=True, postprocessing=sharpen)
    match_type_texts = [ocr.read(i) for i in match_type_imgs]
    return match_type_texts

def infer_match_results(img, classifier, scaler, batch_index=0, bgr=True):
    img = load_img(img, bgr=bgr)
    match_result_imgs = extract(img, "MATCH_RESULT_LIST", scaler=scaler, batch_index=batch_index%4, split_list=True, crop_list=True)
    match_result_classes = classifier.infer(match_result_imgs)
    return match_result_classes

def generate_mask(match_types, match_results, similarity=DEFAULT_SIMILARITY):
    assert len(match_types) == len(match_results)
    match_results_mask = [c != "Invalid" for c in match_results]
    match_types_mask = [similarity(s, "ranked") >= 0.8 for s in match_types]
    final_mask = [(match_results_mask[i] and match_types_mask) for i in range(len(match_types))]
    return final_mask

def filter_batch(ss_batch, mask):
    return [ss_batch[i+1] for i in range(len(mask)) if mask[i]]

def generate_cp(ss_batch, input_dir, output_dir, player_name):
    return _generate_mv(ss_batch, input_dir, output_dir, player_name, concat_input=True)

class Filterer:
    def __init__(self, input_dir, output_dir, ocr=None, classifier=None, scaler=None, img=None, batch_size=BATCH_SIZE):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        assert scaler or img
        scaler = scaler or Scaler(img)
        self.scaler = scaler
        self.ocr = ocr or OCR(has_number=False)
        self.classifier = classifier or MatchResultListClassifier()

    def output_dir_player(self, player_name):
        return os.path.join(self.output_dir, player_name)

    def input_dir_player(self, player_name):
        return os.path.join(self.input_dir, player_name)

    def create_batches(self, player_name):
        return create_batches(self.input_dir_player(player_name), batch_size=self.batch_size)

    def read_match_types(self, img, batch_index=0, bgr=True):
        return read_match_types(img, self.ocr, self.scaler, batch_index=batch_index%4, bgr=bgr)

    def infer_match_results(self, img, batch_index=0, bgr=True):
        return infer_match_results(img, self.classifier, self.scaler, batch_index=batch_index%4, bgr=bgr)
    
    def generate_cp(self, ss_batch, player_name, batch_index=0):
        img = os.path.join(self.input_dir_player(player_name), ss_batch[0])
        img = load_img(img)
        match_types = self.read_match_types(img, batch_index=batch_index%4, bgr=False)
        match_results = self.read_match_results(img, batch_index=batch_index%4, bgr=False)
        mask = generate_mask(match_types, match_results)
        valid_ss = filter_batch(ss_batch, mask)
        player_input_dir = self.input_dir_player(player_name)
        player_output_dir = self.output_dir_player(player_name)
        cps = _generate_mv(
            valid_ss,
            player_input_dir,
            player_output_dir
        )
        return player_output_dir, cps
    
    def generate_cp_player(self, player_name):
        batches = self.create_batches(player_name)
        cps = [self.generate_cp(batch, i%4) for i, batch in enumerate(batches)]
        return cps

    def generate_cp_all(self):
        players = os.listdir(self.input_dir)
        for player in players:
            yield self.generate_cp_player(player)
