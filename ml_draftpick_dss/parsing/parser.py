import os
from .preprocessing import sharpen, load_img, circle_mask, remove_artifact
from .cropping import extract
from .ocr import OCR, DEFAULT_SIMILARITY
from .scaler import Scaler
from .classifier import MatchResultClassifier, HeroIconClassifier, MedalClassifier, ScreenshotClassifier
from .grouping import infer_ss_type, read_opening_failure, check_opening_failure
from .util import inference_save_path, read_save_path, save_inference, mkdir, exception_message

def read_battle_id(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    battle_id_img = extract(img, "BATTLE_ID", scaler=scaler)
    battle_id_int = ocr.read_battle_id(battle_id_img)
    return battle_id_int, battle_id_img

def infer_match_result(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_result_img = extract(img, "MATCH_RESULT", scaler=scaler)
    match_result_text = classifier.infer([match_result_img])[0]
    return match_result_text, match_result_img

def read_match_duration(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_duration_img = extract(img, "MATCH_DURATION", scaler=scaler)
    match_duration_float = ocr.read_match_duration_mins(match_duration_img)
    return match_duration_float, match_duration_img

def read_team_kills(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    team_kills_imgs = [extract(img, "TEAM_KILLS", scaler=scaler, reverse_x=r) for r in (False, True)]
    team_kills_ints = [ocr.read_int(i) for i in team_kills_imgs]
    return team_kills_ints, team_kills_imgs

def hero_icon_postprocessing(x):
    x = circle_mask(x)
    x = remove_artifact(x)
    return x

def infer_heroes(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    hero_imgs = [extract(img, "HERO_LIST", scaler=scaler, split_list=True, crop_list=True, postprocessing=hero_icon_postprocessing, reverse_x=r) for r in (False, True)]
    hero_classes = [classifier.infer(i) for i in hero_imgs]
    return hero_classes, hero_imgs

def infer_medals(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    medal_imgs = [extract(img, "MEDAL_LIST", scaler=scaler, split_list=True, crop_list=True, reverse_x=r) for r in (False, True)]
    medal_classes = [classifier.infer(i) for i in medal_imgs]
    return medal_classes, medal_imgs

def read_scores(img, ocr, scaler, bgr=True, throw=True):
    img = load_img(img, bgr=bgr)
    score_imgs = [extract(img, "SCORE_LIST", scaler=scaler, split_list=True, crop_list=True, reverse_x=r) for r in (False, True)]
    score_floats = [[ocr.read_score(j, throw=throw) for j in i] for i in score_imgs]
    return score_floats, score_imgs

class Parser:
    def __init__(
            self, input_dir, 
            ss_classifier,
            match_result_classifier, 
            hero_icon_classifier, 
            medal_classifier, 
            ocr=None, img_size=None,
            inference_save_dir="inferences"
        ):
        self.input_dir = input_dir
        assert isinstance(ss_classifier, ScreenshotClassifier)
        assert isinstance(match_result_classifier, MatchResultClassifier)
        assert isinstance(hero_icon_classifier, HeroIconClassifier)
        assert isinstance(medal_classifier, MedalClassifier)
        self.ss_classifier = ss_classifier
        self.match_result_classifier = match_result_classifier
        self.hero_icon_classifier = hero_icon_classifier
        self.medal_classifier = medal_classifier
        self.inference_save_dir = inference_save_dir
        mkdir(self.inference_save_dir)
        self.img_size = img_size
        scaler = Scaler(img_size) if img_size else None
        self._scaler = scaler
        self.scaler = None
        self.ocr = ocr or OCR(has_number=False)

    def input_dir_player(self, player_name):
        return os.path.join(self.input_dir, player_name)

    def inference_save_path(self, feature, infered_class, relpath, index=0):
        return inference_save_path(self.inference_save_dir, feature, infered_class, relpath, index=index)

    def read_save_path(self, feature, read, relpath, index=0):
        return read_save_path(self.inference_save_dir, feature, read, relpath, index=index)
    
    def read_battle_id(self, img, bgr=True):
        return read_battle_id(img, self.ocr, self.scaler, bgr=bgr)

    def infer_match_result(self, img, bgr=True):
        return infer_match_result(img, self.match_result_classifier, self.scaler, bgr=bgr)

    def read_match_duration(self, img,bgr=True):
        return read_match_duration(img, self.ocr, self.scaler, bgr=bgr)

    def read_team_kills(self, img, bgr=True):
        return read_team_kills(img, self.ocr, self.scaler, bgr=bgr)

    def infer_heroes(self, img, bgr=True):
        return infer_heroes(img, self.hero_icon_classifier, self.scaler, bgr=bgr)

    def infer_medals(self, img, bgr=True):
        return infer_medals(img, self.medal_classifier, self.scaler, bgr=bgr)

    def read_scores(self, img, bgr=True):
        return read_scores(img, self.ocr, self.scaler, bgr=bgr)
    
    def input_relpath(self, path):
        return os.path.relpath(path, self.input_dir)

    def infer_ss_type(self, img, bgr=True):
        return infer_ss_type(img, self.ss_classifier, self.scaler, bgr=bgr)

    def read_opening_failure(self, img, bgr=True):
        return read_opening_failure(img, self.ocr, scaler=self.scaler, bgr=bgr)

    def check_opening_failure(self, text, similarity=DEFAULT_SIMILARITY):
        return check_opening_failure(text, similarity=similarity)
    
    def infer(self, ss_path, player_name, throw=False, return_img=False):
        relpath = self.input_relpath(ss_path)
        img = load_img(ss_path, resize=self.img_size)
        self.scaler = self._scaler or Scaler(img)

        ss_type, ss_type_img = self.infer_ss_type(img, bgr=False)
        assert ((not throw) or (ss_type=="Result")), f"HISTORY: {ss_path}"

        opening_failure_text, opening_failure_img = self.read_opening_failure(img, bgr=False)
        opening_failure = self.check_opening_failure(opening_failure_text)
        assert ((not throw) or (not opening_failure)), f"OPENING_FAILURE: {ss_path}"

        match_result, match_result_img = self.infer_match_result(img, bgr=False)
        assert ((not throw) or (match_result != "Invalid")), f"INVALID: {ss_path}"

        medals, medals_img = self.infer_medals(img, bgr=False)
        assert ((not throw) or ("AFK" not in (medals[0] + medals[1]))), f"AFK: {ss_path}; {medals}"

        heroes, heroes_img = self.infer_heroes(img, bgr=False)
        assert ((not throw) or (len(set(heroes[0] + heroes[1])) == 10)), f"DOUBLE: {ss_path}; {heroes}"
        
        try:
            battle_id, battle_id_img = self.read_battle_id(img, bgr=False)
            match_duration, match_duration_img = self.read_match_duration(img, bgr=False)
            team_kills, team_kills_img = self.read_team_kills(img, bgr=False)
            scores, scores_img = self.read_scores(img, bgr=False, throw=throw)
        except Exception as ex:
            print("relpath", relpath)
            raise

        obj = {
            "file": relpath,
            "player": player_name,
            "ss_type": ss_type,
            "opening_failure": opening_failure,
            "battle_id": battle_id,
            "match_result": match_result,
            "match_duration": match_duration,
            "left_team_kills": team_kills[0],
            "right_team_kills": team_kills[1],
            "left_heroes": heroes[0],
            "right_heroes": heroes[1],
            "left_medals": medals[0],
            "right_medals": medals[1],
            "left_scores": scores[0],
            "right_scores": scores[1]
        }
        if return_img:
            obj = {
                **obj,
                "ss_type_img": ss_type_img,
                "opening_failure_img": opening_failure_img,
                "battle_id_img": battle_id_img,
                "match_result_img": match_result_img,
                "match_duration_img": match_duration_img,
                "left_team_kills_img": team_kills_img[0],
                "right_team_kills_img": team_kills_img[1],
                "left_heroes_img": heroes_img[0],
                "right_heroes_img": heroes_img[1],
                "left_medals_img": medals_img[0],
                "right_medals_img": medals_img[1],
                "left_scores_img": scores_img[0],
                "right_scores_img": scores_img[1]
            }
        return obj
    
    def _infer_player(self, player_name, throw=False, return_img=False):
        input_dir_player = self.input_dir_player(player_name)
        files = os.listdir(input_dir_player)
        objs = [self.infer(os.path.join(input_dir_player, file), player_name, throw=throw, return_img=return_img) for file in files]
        return objs
    
    def _infer_player_split(self, player_name, return_img=False):
        input_dir_player = self.input_dir_player(player_name)
        files = os.listdir(input_dir_player)
        valid_objs, history_files, invalid_files, afk_files, double_files = [], [], [], []
        for file in files:
            path = os.path.join(input_dir_player, file)
            relpath = self.input_relpath(path)
            try:
                obj = self.infer(path, player_name, throw=True, return_img=return_img)
                valid_objs.append(obj)
            except AssertionError as ex:
                message = exception_message(ex)
                if message.startswith("HISTORY") or message.startswith("OPENING_FAILURE"):
                    print(message)
                    history_files.append(relpath)
                if message.startswith("INVALID"):
                    print(message)
                    invalid_files.append(relpath)
                elif message.startswith("AFK"):
                    print(message)
                    afk_files.append(relpath)
                elif message.startswith("DOUBLE"):
                    print(message)
                    double_files.append(relpath)
                else:
                    raise
        return valid_objs, history_files, invalid_files, afk_files, double_files
    
    def infer_player(self, player_name, split=True, throw=False, return_img=False):
        if split:
            return self._infer_player_split(player_name, return_img=return_img)
        return self._infer_player(player_name, throw=throw, return_img=return_img)

    def infer_all(self, split=True, throw=False, return_img=False, start=None, exclude={}):
        players = os.listdir(self.input_dir)
        if start:
            players = players[players.index(start):]
        if exclude:
            players = [p for p in players if p not in exclude]
        for player in players:
            print("Infering", player)
            yield self.infer_player(player, split=split, throw=throw, return_img=return_img)

    def save_inference(self, obj):
        for feature in ["ss_type", "match_result", "left_heroes", "right_heroes", "left_medals", "right_medals"]:
            save_inference(obj, self.inference_save_path, feature)
        for feature in ["opening_failure", "battle_id", "match_duration", "left_team_kills", "right_team_kills", "left_scores", "right_scores"]:
            save_inference(obj, self.read_save_path, feature)

def is_invalid(obj):
    return obj["match_result"] == "Invalid"

def has_afk(obj):
    return "AFK" in (obj["medals"][0] + obj["medals"][1])

def filter_invalid(objs, split=True):
    valid = [obj for obj in objs if not is_invalid(obj)]
    if not split:
        return valid
    return (
        valid,
        [obj for obj in objs if is_invalid(obj)]
    )

def filter_afk(objs, split=True):
    valid = [obj for obj in objs if not has_afk(obj)]
    if not split:
        return valid
    return (
        valid,
        [obj for obj in objs if has_afk(obj)]
    )
