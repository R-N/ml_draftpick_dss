from paddleocr import PaddleOCR
from thefuzz import fuzz

def DEFAULT_SIMILARITY(*args, **kwargs):
    return 0.01 * fuzz.token_sort_ratio(*args, **kwargs)

# use en if text contains numbers
# otherwise use latin

class OCR:
    def __init__(self, has_number=True, similarity=DEFAULT_SIMILARITY):
        lang = "en" if has_number else "latin"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        self.similarity = similarity

    def read(self, img):
        text = self.ocr.ocr(
            img, det=False, cls=False
        )[0][0]
        if not text:
            return None
        return text[0].strip().lower()

    def read_history_player_name(self, img):
        name_text = self.read(img)
        name_text = name_text.replace("`", "'").replace('"', "'").rsplit("'", maxsplit=1)
        print(name_text)
        if (not len(name_text) == 2):
            raise Exception("Invalid SS")
        if self.similarity(name_text[-1].rsplit(" ", 2)[-1], "history") < 0.8:
            raise Exception("Invalid SS")
        name_text = name_text[0]
        return name_text
    
    def read_num(self, img):
        num = self.read(img)
        num = num.strip(".").replace("!", "").replace("/", "").replace("%", "")
        return num
    
    def read_int(self, img):
        num = self.read_num(img)
        return int(num.replace(".", ""))

    def read_score(self, img):
        score_text = self.read_num(img)
        try:
            replace = float(score_text) >= 100
            replace = -2 if replace else -1
            score_text = score_text if "." in score_text else (score_text[:replace] + "." + score_text[-1:])
            score_f = float(score_text)
            score_f = score_f if score_f < 70 else (score_f - 60)
        except Exception as ex:
            raise Exception("Invalid SS")
        return score_f
    
    def read_battle_id(self, img):
        text = self.read(img)
        num = int(text.split(" ")[-1].strip())
        return num
    
    def read_match_duration(self, img):
        text = self.read(img)
        time = text.split(" ")[-1].strip()[:5].replace(".", ":")
        return time
    
    def read_match_duration_mins(self, img):
        time = self.read_match_duration(img)
        mins, sec = [int(x) for x in time.split(":")]
        total_mins = mins + sec/60
        return total_mins
