import pandas as pd
from requests_html import HTMLSession
from .util import parse_number, standardize_name
from ..constants import HERO_LIST

HERO_URL = "https://mobile-legends.fandom.com/wiki"


def get_growth_td(row):
    wide = "colspan" in row[1].attrs and row[1].colspan == 2
    return row[2] if wide else row[3]

def has_growth(row):
    return len(row) > 3 and row[3].text.strip() != "-"

def scrap_hero_attr(hero):
    session = HTMLSession()
    r = session.get(f"{HERO_URL}/{hero}")
    table = r.html.find("table.wikitable", first=True)
    rows = table.find("tr")[2:]
    rows = [r.find("td") for r in rows]
    rows = [r for r in rows if len(r)]

    try:
        obj = {standardize_name(row[0].text): row for row in rows}
        obj_growth = {f"{k}_growth": parse_number(row[3].text) for k, row in obj.items() if has_growth(row)}
        obj = {
            "name": hero,
            **{k: parse_number(row[1].text) for k, row in obj.items()},
            **obj_growth
        }
    except (IndexError, ValueError) as ex:
        print("Hero:", hero)
        print("Rows:", len(rows))
        print("Lens:", [len(row) for row in rows])
        return {"name": hero}
        raise ex
    return obj

def scrap_all(heroes=HERO_LIST):
    objs = [scrap_hero_attr(hero) for hero in heroes]
    objs = [obj for obj in objs if obj]
    df = pd.DataFrame.from_dict(objs)
    return df