from requests_html import HTMLSession
import pandas as pd
from .util import parse_multiple

HERO_LIST_URL = 'https://mobile-legends.fandom.com/wiki/List_of_heroes'

def scrap():
    session = HTMLSession()

    r = session.get(HERO_LIST_URL)
    rows = r.html.find('table.wikitable tr')[1:]
    #print(len(rows))

    rowcols = [r.find('td') for r in rows]
    heroes = [
        {
            "id": r[2].text,
            "name": r[1].text,
            "roles": parse_multiple(r[3].text),
            "specialities": parse_multiple(r[4].text),
            "lane": parse_multiple(r[5].text),
            "icon": r[0].find("img", first=True).attrs["data-src"]
        }
        for r in rowcols
    ]

    df = pd.DataFrame(heroes)
    df.set_index("id", inplace=True)
    return df
