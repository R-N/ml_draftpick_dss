import pandas as pd
import torch

def join_tag(df, col, include_na=True):
    if col is None:
        return df
    if include_na:
        df[col] = df[col].fillna("<NA>")
    else:
        df = df.loc[~df[col].isna()]
    tag = pd.DataFrame(df[col].tolist(), index=df.index).stack()
    tag.index = tag.index.droplevel(-1)
    tag.name = col
    # tag = tag.drop_duplicates()
    return df.drop(col, axis=1).join(tag)

def groupby_tag(df, col, include_na=True):
    if col is None:
        return df
    #df.reset_index(level=0, inplace=True)
    return join_tag(df, col, include_na=include_na).groupby(col)
def aggregate(
    df, 
    x, y, 
    y_list=True,
    aggfunc=lambda x: x.sum(), 
    agg=True,
    agg_val=False,
    agg_name="All"
):
    agg_val = agg and agg_val
    y_list = y_list and y is not None
    if agg_val:
        agg_val_val = aggfunc(df)
        agg_val_val[y] = agg_name
    if y_list:
        if agg:
            df = aggfunc(groupby_tag(
                df,
                y
            )).sort_values(x, ascending=True)
            df[y] = df.index
            sorting = df
            #y = None
        else:
            df = join_tag(
                df,
                y
            ).sort_values(x, ascending=True)
            sorting = aggfunc(groupby_tag(df[[x, y]], y))
    elif y is not None:
        sorting = aggfunc(df.groupby(y))
    else:
        sorting = df
    sorting = sorting.sort_values(x, ascending=True)
    if agg_val:
        df = pd.concat([
            df,
            pd.DataFrame([agg_val_val], index=[agg_name])[df.columns]
        ])
    return df, sorting


def sig_to_tanh_range(x):
    return x * 2 - 1

def tanh_to_sig_range(x):
    return (x + 1) / 2.0


def split_dim(x, dim=-1, squeeze=False):
    ret = [y for y in torch.split(x, 1, dim=dim)]
    if squeeze:
        ret = [torch.squeeze(y, dim=dim) for y in ret]
    return ret