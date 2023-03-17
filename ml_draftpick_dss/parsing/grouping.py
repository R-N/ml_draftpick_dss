import os
from .preprocessing import sharpen, load_img
from .cropping import extract
from .ocr import OCR
from .scaler import Scaler

BATCH_SIZE = 1+5

def create_batches(input_dir, batch_size=BATCH_SIZE):
    ss_list = list(sorted(os.listdir(input_dir)))
    ss_list = [s for s in ss_list if not s.endswith(".ini")]
    n = len(ss_list)
    n = n // batch_size + (1 if n % batch_size > 0 else 0)
    ss_groups = [ss_list[i*batch_size:i*batch_size+batch_size] for i in range(n)]
    return ss_groups

def read_player_name(img, ocr, scaler, batch_index=0, bgr=True):
    img = load_img(img, bgr=bgr)
    name_img = extract(img, "HISTORY_PLAYER_NAME", scaler=scaler, postprocessing=sharpen, batch_index=batch_index%4)
    name_text = ocr.read_history_player_name(name_img)
    return name_text

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
    def __init__(self, input_dir, output_dir, ocr=None, scaler=None, img=None, batch_size=BATCH_SIZE):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        assert scaler or img
        scaler = scaler or Scaler(img)
        self.scaler = scaler
        self.ocr = ocr or OCR(has_number=False)

    def output_dir_player(self, player_name):
        return os.path.join(self.output_dir, player_name)

    def create_batches(self):
        return create_batches(self.input_dir, batch_size=self.batch_size)

    def read_player_name(self, img, batch_index=0, bgr=True):
        return read_player_name(img, self.ocr, self.scaler, batch_index=batch_index%4, bgr=bgr)
    
    def generate_mv(self, ss_batch, batch_index=0):
        player_name = self.read_player_name(
            os.path.join(self.input_dir, ss_batch[0]), 
            batch_index=batch_index%4
        )
        player_output_dir = self.output_dir_player(player_name)
        mvs = generate_mv(
            ss_batch,
            self.input_dir,
            player_output_dir
        )
        return player_output_dir, mvs
    
    def generate_groups(self):
        batches = self.create_batches()
        for i, batch in enumerate(batches):
            yield(self.generate_mv(batch, i%4))
