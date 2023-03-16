import pandas as pd
from requests_html import HTMLSession
from .hero_attributes import HERO_URL
from .util import standardize_name

def parse_skill(hero, table):
    rows = table.find(":root>tbody>tr")
    desc = rows[0].find("td")[1]
    desc_divs = desc.find("div")
    skill_name = desc_divs[0].text
    if len(desc_divs) > 1:
        skill_tags = desc_divs[1].find("span")
        skill_tags = ",".join([t.text for t in skill_tags])
        skill_type = desc_divs[1].text
        skill_type = skill_type.split("Type:")[-1].strip() if ":" in skill_type else None
    else:
        skill_tags = None
        skill_type = None
    desc = desc.text

    obj = {
        "hero": hero,
        "name": skill_name,
        "tags": skill_tags,
        "type": skill_type,
        "desc": desc
    }

    details = rows[-1] if len(rows) > 1 else None
    if not details:
        return obj
    details = details.find("table")
    details = details[0] if details else None
    if not details:
        return obj
    details = details.find("tr")
    if not details:
        return obj
    details = [row.find("td") for row in details]
    details = [row for row in details if row and len(row) > 1 and [td.text for td in row]]
    if not details:
        return obj
    details = {standardize_name(row[0].text): row[1].text.strip() for row in details}
    obj = {**obj, **details}
    return obj

def scrap_skills(hero, url):
    session = HTMLSession()
    r = session.get(url)

    tables = r.html.find("table.wikitable tbody")
    tables = [t for t in tables if len(t.find('td[width="100px"] img'))]
    
    skills = [parse_skill(hero, table) for table in tables]
    return skills

def scrap_hero_skills(hero):
    url = f"{HERO_URL}/{hero}"

    #print("Scraping", hero)
    skills = scrap_skills(hero, url)

    if not skills:
        url = f"{HERO_URL}/{hero}/Abilities"
        skills = scrap_skills(hero, url)

    if not skills:
        raise Exception(f"Unable to find skills for {hero}")

    return skills

def scrap_all(heroes):
    hero_skills = [scrap_hero_skills(hero) for hero in heroes]
    hero_skills = [skills for skills in hero_skills if skills]
    hero_skills = [[skill for skill in skills if skill] for skills in hero_skills]
    #len(hero_skills)
    hero_skills = [skill for skills in hero_skills for skill in skills]
    df = pd.DataFrame.from_dict(hero_skills)
    return df
