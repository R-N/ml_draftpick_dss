from .scaler import Scaler
from skimage.util import crop as _crop

SS_CORNER = (
    (0, 2050),
    (0, 1650)
)
OPENING_FAILURE = (
    (470, 540),
    (640, 640)
)
HISTORY_PLAYER_NAME = (
    (10, 2070), 
    (260, 500)
)
MATCH_TYPE_LIST = (
    (120, 0), 
    (1550, 120)
)

MATCH_TYPE_LIST_CROP = lambda batch_index: (
    (66 - 13*batch_index, 84 + 13*batch_index), 
    (30, 30)
)

MATCH_RESULT_LIST = (
    (120, 0),
    (625, 1050)
)

MATCH_RESULT_LIST_CROP = lambda batch_index: (
    (42 - 13*batch_index, 36 + 13*batch_index),
    (0, 0)
)

HERO_LIST = (
    (255, 125),
    (168, 1578)
)

HERO_LIST_CROP = (
    (19, 26),
    (34, 38)
)

BATTLE_ID = (
    (945, 90),
    (32, 1480)
)

MATCH_RESULT = (
    (20, 2050),
    (750, 750)
)
TEAM_KILLS = (
    (30, 2025),
    (450, 1200)
)
MATCH_DURATION = (
    (160, 880),
    (1420, 100)
)
SCORE_LIST = (
    HERO_LIST[0],
    (825, 970)
)
SCORE_LIST_CROP = (
    (93, 16),
    (28, 28)
)
MEDAL_LIST = SCORE_LIST
MEDAL_LIST_CROP = (
    (8, 50),
    (4, 4)
)

CROPPINGS = {
    "SS_CORNER": SS_CORNER,
    "OPENING_FAILURE": OPENING_FAILURE,
    "HISTORY_PLAYER_NAME": HISTORY_PLAYER_NAME,
    "MATCH_TYPE_LIST": MATCH_TYPE_LIST,
    "MATCH_TYPE_LIST_CROP": MATCH_TYPE_LIST_CROP,
    "MATCH_RESULT_LIST": MATCH_RESULT_LIST,
    "MATCH_RESULT_LIST_CROP": MATCH_RESULT_LIST_CROP,
    "HERO_LIST": HERO_LIST,
    "HERO_LIST_CROP": HERO_LIST_CROP,
    "BATTLE_ID": BATTLE_ID,
    "MATCH_RESULT": MATCH_RESULT,
    "TEAM_KILLS": TEAM_KILLS,
    "MATCH_DURATION": MATCH_DURATION,
    "SCORE_LIST": SCORE_LIST,
    "SCORE_LIST_CROP": SCORE_LIST_CROP,
    "MEDAL_LIST": MEDAL_LIST,
    "MEDAL_LIST_CROP": MEDAL_LIST_CROP,
}
OFFSETS = {
    "SS_CORNER": 0.4,
    "HISTORY_PLAYER_NAME": 0.4,
    "MATCH_TYPE_LIST": 0.4,
    "MATCH_RESULT_LIST": 0.4,
    "HERO_LIST": 0.4,
    "TEAM_KILLS": 0.2,
    "SCORE_LIST": 0.2,
    "MEDAL_LIST": 0.2,
}

def get_cropping(name, scaler, batch_index=0, offset=0, reverse_x=False):
    cropping = CROPPINGS[name]
    if callable(cropping):
        cropping = cropping(batch_index)
    if name in OFFSETS:
        offset += OFFSETS[name]
    x = "CROP" not in name
    cropping = (
        scaler.scale(cropping[0]),
        scaler.scale(cropping[1], x=x, offset=offset)
    )
    if reverse_x:
        cropping = (cropping[0], reversed(cropping[1]))
    return cropping

def crop(img, cropping):
    return _crop(img, [*cropping, (0, 0)])

def get_ith(img_list, i):
    img_list_h = img_list.shape[0]
    img_h = img_list_h // 5
    img_up = img_h * i
    img_down = img_h * (5-1-i)
    img_i = crop(img_list, [(img_up, img_down), (0, 0)])
    return img_i

def extract(img, name, batch_index=0, offset=0, scaler=None, reverse_x=False, split_list=False, crop_list=False, postprocessing=None):
    scaler = scaler or Scaler(img)
    cropping = get_cropping(name, scaler, batch_index=batch_index, offset=offset, reverse_x=reverse_x)
    result = crop(img, cropping)
    if split_list and "LIST" in name:
        result = [get_ith(result, i) for i in range(0, 5)]
        if crop_list:
            result = [extract(r, f"{name}_CROP", batch_index=batch_index, offset=offset, scaler=scaler) for r in result]
    if postprocessing:
        result = postprocess(result, postprocessing)
    return result

def postprocess(img, postprocessing):
    if isinstance(img, list):
        result = [postprocessing(i) for i in img]
    else:
        result = postprocessing(img)
    return result
