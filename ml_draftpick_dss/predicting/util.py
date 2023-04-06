import pandas as pd
import torch
from ..util import mkdir
from math import sqrt
import scipy.stats as st

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

def get_unique(mixed):
    return sorted(list(set(mixed)), key=lambda x: (x is None, x))
def get_basic_c(c):
    return c.split("_", maxsplit=1)[0]

def progressive_smooth(last, weight, point):
    return last * weight + (1 - weight) * point

def calculate_prediction_interval(series, alpha=0.05, n=None):
    n = (n or len(series))
    mean = sum(series) / max(1, n)
    sum_err = sum([(mean - x)**2 for x in series])
    stdev = sqrt(1 / max(1, n - 2) * sum_err)
    mul = st.norm.ppf(1.0 - alpha) if alpha >= 0 else 2 + alpha
    sigma = mul * stdev
    return mean, sigma


def round_digits(x, n_digits=0):
    if x is None:
        return x
    x = Decimal(x).as_tuple()
    n = min(len(x.digits), n_digits + 2)
    digits = sum([x.digits[i] * 10**(-i) for i in range(n)])
    if n_digits == 0:
        digits = floor(digits)
    else:
        digits = round(digits, n_digits)
    exp = x.exponent + len(x.digits) - 1
    return digits * 10**exp