import pandas as pd

MULTIPLE_ATTRS = ["roles", "specialities"]
def load_heroes(heroes_path):
    df_heroes = pd.read_csv(heroes_path)
    for x in MULTIPLE_ATTRS:
        df_heroes[x] = df_heroes[x].str.split(",")
        df_heroes[[f"{x}_{i}" for i in range(2)]] = pd.DataFrame(df_heroes[x].tolist(), index=df_heroes.index)


    return df_heroes