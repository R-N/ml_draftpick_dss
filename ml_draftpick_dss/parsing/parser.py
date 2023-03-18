import os
from .preprocessing import sharpen, load_img, circle_mask, remove_artifact
from .cropping import extract
from .ocr import OCR
from .scaler import Scaler
from .classifier import MatchResultClassifier, HeroIconClassifier, MedalClassifier
from .util import inference_save_path, read_save_path, save_inference

def read_battle_id(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    battle_id_img = extract(img, "BATTLE_ID", scaler=scaler, postprocessing=sharpen)
    battle_id_int = ocr.read_battle_id(battle_id_img)
    return battle_id_int, battle_id_img

def infer_match_result(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_result_img = extract(img, "MATCH_RESULT", scaler=scaler)
    match_result_text = classifier.infer([match_result_img])[0]
    return match_result_text, match_result_img

def read_match_duration(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_duration_img = extract(img, "MATCH_DURATION", scaler=scaler, postprocessing=sharpen)
    match_duration_float = ocr.read_match_duration_mins(match_duration_img)
    return match_duration_float, match_duration_img

def read_team_kills(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    team_kills_imgs = [extract(img, "TEAM_KILLS", scaler=scaler, postprocessing=sharpen, reverse_x=r) for r in (False, True)]
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

def read_scores(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    score_imgs = [extract(img, "SCORE_LIST", scaler=scaler, split_list=True, crop_list=True, reverse_x=r) for r in (False, True)]
    score_floats = [ocr.read_score(i) for i in score_imgs]
    return score_floats, score_imgs

class Parser:
    def __init__(
            self, input_dir, ocr=None, 
            match_result_classifier=None, 
            hero_icon_classifier=None, 
            medal_classifier=None, 
            scaler=None, img=None
        ):
        self.input_dir = input_dir
        assert scaler or img
        scaler = scaler or Scaler(img)
        self.scaler = scaler
        self.ocr = ocr or OCR(has_number=False)
        self.match_result_classifier = match_result_classifier or MatchResultClassifier()
        self.hero_icon_classifier = hero_icon_classifier or HeroIconClassifier()
        self.medal_classifier = medal_classifier or MedalClassifier()

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
    
    def infer(self, ss_path, player_name, throw=False, return_img=False):
        img = load_img(ss_path)

        match_result, match_result_img = self.infer_match_result(img, bgr=False)
        assert ((not throw) or match_result != "Invalid"), f"INVALID: {ss_path}"

        medals, medals_img = self.infer_medals(img, bgr=False)
        assert ((not throw) or len([m for m in medals if m == "AFK"])) == 0, f"AFK: {ss_path}; {medals}"

        heroes, heroes_img = self.infer_heroes(img, bgr=False)
        assert (len(set(heroes[0] + heroes[1])) == 10), f"DOUBLE: {ss_path}; {heroes}"
        
        relpath = self.input_relpath(ss_path)
        try:
            battle_id, battle_id_img = self.read_battle_id(img, bgr=False)
            match_duration, match_duration_img = self.read_match_duration(img, bgr=False)
            team_kills, team_kills_img = self.read_team_kills(img, bgr=False)
            scores, scores_img = self.read_scores(img, bgr=False)
        except Exception as ex:
            print("relpath", relpath)
            raise

        obj = {
            "file": relpath,
            "player": player_name,
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
        valid_objs, invalid_files, afk_files, double_files = [], [], [], []
        for file in files:
            path = os.path.join(input_dir_player, file)
            relpath = self.input_relpath(path)
            try:
                obj = self.infer(path, player_name, throw=True, return_img=return_img)
                valid_objs.append(obj)
            except AssertionError as ex:
                message = ex.message if hasattr(ex, "message") else str(ex)
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
        return valid_objs, invalid_files, afk_files, double_files
    
    def infer_player(self, player_name, split=True, throw=False, return_img=False):
        if split:
            return self._infer_player_split(player_name, return_img=return_img)
        return self._infer_player(player_name, throw=throw, return_img=return_img)

    def infer_all(self, split=True, throw=False, return_img=False):
        players = os.listdir(self.input_dir)
        for player in players:
            yield self.infer_player(player, split=split, throw=throw, return_img=return_img)

    def save_inference(self, obj):
        for feature in ["match_result", "left_heroes", "right_heroes", "left_medals", "right_medals"]:
            save_inference(obj, self.inference_save_path, feature)
        for feature in ["battle_id", "match_duration", "left_team_kills", "right_team_kills", "left_scores", "right_scores"]:
            save_inference(obj, self.read_save_path, feature)

def is_invalid(obj):
    return obj["match_result"] == "Invalid"

def has_afk(obj):
    return len([m for m in obj["medals"] if m == "AFK"]) > 0

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
