
import pandas as pd
from ast import literal_eval
DUAL_ATTRS =  ["heroes", "medals", "scores"]
FLIP_COLS = ["team_kills", "heroes", "medals", "scores", "scores_sum"]
LRS = ["left", "right"]
FLIP_LRS = {"left": "right", "right": "left"}
DIFF_SUM = ["diff", "sum"]
DIFF_SUM_OPS = {
    "diff": (lambda x, y: x-y), 
    "sum": (lambda x, y: x+y)
}
DIFF_SUM_COLS = ["scores_sum", "team_kills"]
DIFF_COLS = [f"{attr}_diff" for attr in DIFF_SUM_COLS]
NORMALIZE_COLS = ["scores_sum_diff", "match_duration", "team_kills_diff"]
NORMALIZED_COLS = [f"{x}_norm" for x in NORMALIZE_COLS]
ADD_NEGATIVE_FLIP_COLS = [
    "objective", 
    "left_victory", 
    *[x for x in NORMALIZE_COLS if "diff" in x],
    #"match_duration",
    *[x for x in NORMALIZED_COLS if "diff" in x],
]
def flip_result(row):
    ret = {
        **row,
        **{f"{FLIP_LRS[lr]}_{col}": row[f"{lr}_{col}"] for lr in LRS for col in FLIP_COLS},
        #**{attr: -row[attr] for attr in DIFF_COLS},
        "match_result": "Defeat" if row["match_result"] == "Victory" else "Victory",
        **{attr: -row[attr] for attr in ADD_NEGATIVE_FLIP_COLS if attr in row}
    }
    return ret
def flip_results(df):
    return df.apply(flip_result, axis=1, result_type="expand")
def merge_results(dfs):
    double_df = pd.concat(dfs)
    double_df.reset_index(inplace=True)
    double_df = double_df.drop_duplicates(
        subset = ['battle_id', "match_result"],
        keep='first'
    )
    double_df.set_index("battle_id", inplace=True)
    #print(len(double_df), len(double_df.columns))
    #double_df.tail()
    return double_df
def filter_victory(double_df):
    return double_df[double_df["match_result"] == "Victory"]
def load_results(result_path):
    list_cols = [f"{lr}_{attr}" for attr in DUAL_ATTRS for lr in ("left", "right")]
    score_cols = [x for x in list_cols if "score" in x]
    df = pd.read_csv(result_path, converters={c: literal_eval for c in list_cols})
    df[DUAL_ATTRS] = df.apply(
        lambda row: [row[f"left_{attr}"]+row[f"right_{attr}"] for attr in DUAL_ATTRS], 
        axis=1,
        result_type="expand"
    )
    df[[f"{attr}_sum" for attr in score_cols]] = df.apply(
        lambda row: [sum(row[attr]) for attr in score_cols],
        axis=1,
        result_type="expand"
    )
    df[[f"{attr}_{ds}" for attr in DIFF_SUM_COLS for ds in DIFF_SUM]] = df.apply(
        lambda row: [
            DIFF_SUM_OPS[ds](row[f"left_{attr}"], row[f"right_{attr}"]) 
            for attr in DIFF_SUM_COLS 
            for ds in DIFF_SUM
        ],
        axis=1,
        result_type="expand"
    )
    """
    for attr in dual_attrs:
        for lr in ("left", "right"):
            attr_lr = f"{lr}_{attr}"
            df[[f"{attr_lr}_{i}" for i in range(5)]] = pd.DataFrame(df[attr_lr].tolist(), index=df.index)
    """
    df["count"] = 1
    df["left_victory"] = (df["match_result"] == "Victory").replace({True: 1, False: -1})
    del df[df.columns[0]]
    df.drop_duplicates(
    subset = ['battle_id'],
    keep='last',
    inplace=True
    )
    df.set_index("battle_id", inplace=True)
    #print(len(df), len(df.columns))
    #df.tail()
    return df
def load_victory(results_path):
    df = load_results(results_path)
    double_df = merge_results(df, flip_results(df))
    victory_df = filter_victory(double_df)
    return victory_df