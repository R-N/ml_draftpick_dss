from paddleocr import PaddleOCR
from thefuzz import fuzz
from .util import show_imgs

def DEFAULT_SIMILARITY(*args, **kwargs):
    return 0.01 * fuzz.token_sort_ratio(*args, **kwargs)

# use en if text contains numbers
# otherwise use latin

class OCR:
    def __init__(self, has_number=True, similarity=DEFAULT_SIMILARITY, batched=False):
        lang = "en" if has_number else "latin"
        self.ocr = PaddleOCR(
            lang=lang,
            ocr_version="PP-OCRv4",
            #rec_model_dir="PaddleOCR_pub/inference/en_PP-OCRv4_rec_infer/",
            #rec_algorithm="SVTR_LCNet",
            drop_score=0.2,
            use_angle_cls=True,
        )
        self.similarity = similarity
        self.batched = batched

    def read(self, img):
        if isinstance(img, list) and not self.batched:
            return [self.read([i])[0] for i in img]
        text = self.ocr.ocr(
            img, det=False, cls=False
        )
        text = [t[0] for t in text]
        text = [t[0] if t else None for t in text]
        return [t.strip() for t in text]

    def read_history_player_name(self, img, throw=True):
        try:
            return self.process_history_player_name(self.read(img))
        except Exception as ex:
            if throw:
                raise
            return None
        
    def process_history_player_name(self, name_text):
        name_text = name_text.replace("`", "'").replace('"', "'").rsplit("'", maxsplit=1)
        if (not len(name_text) == 2):
            raise AssertionError("BAD_SS_HISTORY")
        if self.similarity(name_text[-1].rsplit(" ", 2)[-1], "history") < 0.8:
            raise AssertionError("BAD_SS_HISTORY")
        name_text = name_text[0]
        return name_text
    
    def read_num(self, img):
        return self.process_num(self.read(img))

    def process_num(self, num):
        if isinstance(num, list):
            return [self.process_num(n) for n in num]
        num = num.replace("!", ".")
        num = num.replace("%", "")
        num = num.replace("V", "")
        num = num.replace("\\", "")
        num = num.replace(")", "")
        num = num.replace("G", "6")
        num = num.replace("g", "9")
        num = num.replace("U", "7")
        num = num.replace("T", "7")
        num = num.replace("T", "7")
        num = num.strip(".")
        if num.endswith("/"):
            num = num[:-1] + "1"
        num = num.replace("/", ".") if "." not in num else num.replace("/", "")
        num = num.strip(".")
        return num
    
    def read_int(self, img):
        return self.process_int(self.read(img))

    def process_int(self, num):
        if isinstance(num, list):
            return [self.process_int(n) for n in num]
        num = self.process_num(num)
        return int(num.replace(".", ""))
    
    def read_team_kills(self, img, throw=True):
        try:
            return self.read_int(img)
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_KILL: {ex}")
            return None

    def read_score(self, img, throw=True):
        try:
            return self.process_score(self.read(img))
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_SCORE: {ex}")
            return None
        except AssertionError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_SCORE: {ex}")
            return None
    
    def process_score(self, score_text):
        if isinstance(score_text, list):
            return [self.process_score(n) for n in score_text]
        score_text = self.process_num(score_text)
        if score_text in {"4v", "4v1"}:
            return 4.1
        score_text = score_text.replace("t", "7")
        score_text = score_text.replace("v", ".") if "." not in score_text else score_text.replace("v", "")
        assert score_text.count(".") <= 1, "MULTIPLE_DOTS: {score_text}"
        replace = float(score_text) >= 100
        replace = -2 if replace else -1
        score_text = score_text if "." in score_text else (score_text[:replace] + "." + score_text[-1:])
        score_f = float(score_text)
        score_i = int(score_f // 1)
        if score_i in {77, 79}:
            score_f -= 70
        if score_i in {46}:
            score_f -= 40
        if score_i in {40, 41, 42}:
            score_f -= 30
        if score_i in {17, 18, 19}:
            score_f -= 10
        if score_i in {1}:
            score_f += 6
        if score_f < 3.0:
            score_f = score_f * 10 + 0.1 # This is imprecise by the decimal point but they shouldn't matter much
        score_f = round(score_f, 1)
        #assert (3.0 <= score_f and score_f < 17.0), f"OUTLIER_SCORE: {score_text}, {score_f}"
        return score_f

    def read_battle_id(self, img, throw=True):
        try:
            return self.process_battle_id(self.read(img))
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_BATTLE_ID: {ex}")
            return None

    def process_battle_id(self, text):
        if isinstance(text, list):
            return [self.process_battle_id(n) for n in text]
        text = text.lower().strip()
        text = text.split(":", maxsplit=1)[-1].strip()
        text = text.split(" ", maxsplit=1)[-1].strip()
        if text.startswith("batt"):
            text = text[8:]
        text = text.strip(":").strip()
        return int(text)
    
    def read_match_duration(self, img):
        return self.process_match_duration(self.read(img))
    
    def process_match_duration(self, text):
        if isinstance(text, list):
            return [self.process_match_duration(n) for n in text]
        try:
            text = text.strip()
            text = text.rsplit(" ", maxsplit=1)[-1].strip()
        except ValueError as ex:
            pass
        time = text.strip()[-5:]
        if not time[0].isdigit():
            time = time[1:]
        time = time.replace(".", ":").replace("!", ":").replace(";", ":").strip()
        if ":" not in time:
            time = time.replace(" ", ":")
        return time

    def read_match_duration_mins(self, img, throw=True):
        try:
            return self.process_match_duration_mins(self.read(img))
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_DURATION: {time}; {ex}")
            return None

    def process_match_duration_mins(self, text):
        if isinstance(text, list):
            return [self.process_match_duration_mins(n) for n in text]
        time = self.process_match_duration(text)
        if ":" in time:
            mins, sec = [int(x.strip()) for x in time.split(":")]
        elif len(time) == 4 and time.isnumeric():
            mins, sec = int(time[:2]), int(time[-2:])
        else:
            raise ValueError(f"INVALID_TIME: {time}")
        total_mins = mins + sec/60
        return total_mins
