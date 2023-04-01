import numpy as np
from .heroes_loader import MULTIPLE_ATTRS
import torch
from sklearn import preprocessing
import pandas as pd
from .util import get_mixed, get_basic_c


def get_mixed(df_heroes, x, n=2):
    return [a for i in range(n) for a in df_heroes[f"{x}_{i}"].tolist()]

def encode_batch(f, batch):
    if isinstance(batch, pd.DataFrame) or isinstance(batch, pd.Series):
        batch = batch.tolist()
    if not (torch.is_tensor(batch) or isinstance(batch, np.ndarray)):
        batch = np.array(batch)
    dim = len(batch.shape)
    assert (dim < 3), f"Invalid batch dim: {dim}"
    if dim == 1:
        encoded = [f(hero) for hero in batch]
    elif dim == 2:
        encoded = [[f(hero) for hero in team] for team in batch]
    encoded_tensor = torch.IntTensor(encoded)
    return encoded_tensor

class HeroLabelEncoder:
    def __init__(self, df_heroes):
        mixeds = {x: get_mixed(df_heroes, x) for x in MULTIPLE_ATTRS}
        uniques = {x: get_unique(m) for x, m in mixeds.items()}
        uniques["lane"] = get_unique(df_heroes["lane"])
        uniques["id"] = df_heroes["id"]
        uniques["name"] = df_heroes["name"]

        cols = ["id", "lane", *[f"roles_{i}" for i in range(2)], *[f"specialities_{i}" for i in range(2)]]
        df_heroes_x = df_heroes[cols]
        
        encoders = {c:preprocessing.LabelEncoder().fit(uniques[c]) for c in HERO_COLS}

        df_heroes_x2 = pd.DataFrame(
            {c: encoders[get_basic_c(c)].transform(df_heroes_x[c]) 
            for c in df_heroes_x.columns
        })
        cols2 = ["name", *df_heroes_x2.columns]
        df_heroes_x2["name"] = df_heroes["name"]
        df_heroes_x2 = df_heroes_x2[cols2]
        #df_heroes_x2.head()
        encoding = dict(df_heroes_x2.apply(lambda x: (x["name"], x[df_heroes_x2.columns[1:]].tolist()), axis=1).tolist())
         
        self.encoding = encoding
        self.encoders = encoders
        self.x = df_heroes_x2
        self.mixeds = mixeds
        self.uniques = uniques
        self.dim = len(df_heroes_x.columns)

    def get_encoding(self, hero):
        return self.encoding[hero]
    
    def encode_batch(self, batch):
        return encode_batch(self.get_encoding, batch)
    
    def __call__(self, batch):
        return self.encode_batch(batch)

class HeroOneHotEncoder:
    def __init__(self, df_heroes, include_name=True):

        mixeds = {x: get_mixed(df_heroes, x) for x in MULTIPLE_ATTRS}
        uniques = {x: get_unique(m) for x, m in mixeds.items()}
        uniques["lane"] = get_unique(df_heroes["lane"])
        uniques["id"] = df_heroes["id"]
        uniques["name"] = df_heroes["name"]

        cols = ["name", "lane", *[f"roles_{i}" for i in range(2)], *[f"specialities_{i}" for i in range(2)]]
        df_heroes_x = df_heroes[cols]

        if not include_name:
            cols = cols[1:]

        categories=[uniques[get_basic_c(c)] for c in cols]
        encoder = preprocessing.OneHotEncoder(
            categories=categories,
            sparse_output=False
        ).fit(df_heroes_x[cols])
        encoded = encoder.transform(df_heroes_x[cols])
        dim = encoded.shape[-1]
        dims = [len(c) for c in categories]
        slices = []
        prev = 0
        for d in dims:
            slices.append((prev, prev+d))
            prev += d
        df_encoded = pd.DataFrame(encoded, index=df_heroes["name"])
        encoding = {hero: df_encoded.loc[hero] for hero in uniques["name"]}

        self.mixeds = mixeds
        self.uniques = uniques
        self.encoder = encoder
        self.x = df_encoded
        self.encoding = encoding
        self.dim = dim
        self.dims = dims
        self.slices = slices

    def get_encoding(self, hero):
        return self.encoding[hero]
    
    def encode_batch(self, batch):
        return encode_batch(self.get_encoding, batch)
    
    def __call__(self, batch):
        return self.encode_batch(batch)

