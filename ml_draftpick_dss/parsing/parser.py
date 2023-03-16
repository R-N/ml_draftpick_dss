import os
from .preprocessing import sharpen, load_img
from .cropping import extract
from .ocr import OCR
from .scaler import Scaler
from .classifier import MatchResultClassifier, HeroIconClassifier, MedalClassifier

def read_battle_id(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    battle_id_img = extract(img, "BATTLE_ID", scaler=scaler, postprocessing=sharpen)
    battle_id_int = ocr.read_battle_id(battle_id_img)
    return battle_id_int

def infer_match_result(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_result_img = extract(img, "MATCH_RESULT", scaler=scaler)
    match_result_text = classifier.infer([match_result_img])
    return match_result_text

def read_match_duration(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    match_duration_img = extract(img, "MATCH_DURATION", scaler=scaler, postprocessing=sharpen)
    match_duration_float = ocr.read_match_duration_mins(match_duration_img)
    return match_duration_float

def read_team_kills(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    team_kills_imgs = [extract(img, "TEAM_KILLS", scaler=scaler, postprocessing=sharpen, reverse_x=r) for r in (False, True)]
    team_kills_ints = [ocr.read_int(i) for i in team_kills_imgs]
    return team_kills_ints

def infer_heroes(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    hero_imgs = [extract(img, "HERO_LIST", scaler=scaler, reverse_x=r) for r in (False, True)]
    hero_classes = [classifier.infer(i) for i in hero_imgs]
    return hero_classes

def infer_medals(img, classifier, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    medal_imgs = [extract(img, "MEDAL_LIST", scaler=scaler, reverse_x=r) for r in (False, True)]
    medal_classes = [classifier.infer(i) for i in medal_imgs]
    return medal_classes

def read_scores(img, ocr, scaler, bgr=True):
    img = load_img(img, bgr=bgr)
    score_imgs = [extract(img, "SCORE_LIST", scaler=scaler, postprocessing=sharpen, reverse_x=r) for r in (False, True)]
    score_floats = [ocr.read_score(i) for i in score_imgs]
    return score_floats

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
    
    def parse_match_result(self, ss_path, player_name):
        img = load_img(ss_path)
        
        battle_id = self.read_battle_id(img, bgr=False)
        match_result = self.infer_match_result(img, bgr=False)
        match_duration = self.read_match_duration(img, bgr=False)
        team_kills = self.read_team_kills(img, bgr=False)
        heroes = self.infer_heroes(img, bgr=False)
        medals = self.infer_medals(img, bgr=False)
        scores = self.read_scores(img, bgr=False)

        obj = {
            "file": ss_path,
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
        return obj
    
    def parse_match_result_player(self, player_name):
        files = os.listdir(os.path.join(self.input_dir, player_name))
        objs = [self.parse_match_result(file, player_name) for file in files]
        return objs

    def parse_match_result_all(self):
        players = os.listdir(self.input_dir)
        for player in players:
            yield self.parse_match_result_player(player)
