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

    def read_history_player_name(self, img, throw=True):
        try:
            name_text = self.read(img)
            name_text = name_text.replace("`", "'").replace('"', "'").rsplit("'", maxsplit=1)
            if (not len(name_text) == 2):
                raise AssertionError("BAD_SS_HISTORY")
            if self.similarity(name_text[-1].rsplit(" ", 2)[-1], "history") < 0.8:
                raise AssertionError("BAD_SS_HISTORY")
            name_text = name_text[0]
            return name_text
        except Exception as ex:
            if throw:
                raise
            return None
    
    def read_num(self, img):
        num = self.read(img)
        num = num.replace("!", ".").replace("%", "").strip(".")
        if num.endswith("/"):
            num = num[:-1] + "1"
        num = num.replace("/", ".") if "." not in num else num.replace("/", "")
        num = num.strip(".")
        return num
    
    def read_int(self, img):
        num = self.read_num(img)
        return int(num.replace(".", ""))
    
    def read_team_kills(self, img, throw=True):
        try:
            return self.read_int(img)
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_KILL: {ex}")
            return None

    def read_score(self, img, throw=True):
        score_text = self.read_num(img)
        if score_text in {"4v", "4v1"}:
            return 4.1
        score_text = score_text.replace("t", "7")
        score_text = score_text.replace("v", ".") if "." not in score_text else score_text.replace("v", "")
        try:
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
            if score_f < 3.0:
                score_f = score_f * 10 + 0.1 # This is imprecise by the decimal point but they shouldn't matter much
            score_f = round(score_f, 1)
            #assert (3.0 <= score_f and score_f < 17.0), f"OUTLIER_SCORE: {score_text}, {score_f}"
            return score_f
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_SCORE: {ex}")
            return None
        except AssertionError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_SCORE: {ex}")
            return None
    
    def read_battle_id(self, img, throw=True):
        text = self.read(img)
        try:
            return int(text.split(" ")[-1].strip().strip(":").strip())
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_BATTLE_ID: {ex}")
            return None
    
    def read_match_duration(self, img):
        text = self.read(img)
        try:
            text = text.rsplit(" ", maxsplit=1)[-1]
        except ValueError as ex:
            pass
        time = text.strip()[:5].replace(".", ":").replace("!", ":").replace(";", ":").strip()
        if ":" not in time:
            time = time.replace(" ", ":")
        return time
    
    def read_match_duration_mins(self, img, throw=True):
        time = self.read_match_duration(img)
        try:
            if ":" in time:
                mins, sec = [int(x.strip()) for x in time.split(":")]
            elif len(time) == 4 and time.isnumeric():
                mins, sec = int(time[:2]), int(time[2:])
            else:
                raise ValueError("INVALID_TIME")
            total_mins = mins + sec/60
            return total_mins
        except ValueError as ex:
            if throw:
                raise AssertionError(f"BAD_SS_DURATION: {time}; {ex}")
            return None
