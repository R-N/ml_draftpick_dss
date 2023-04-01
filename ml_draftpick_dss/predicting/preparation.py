import numpy as np
from .result_loader import NORMALIZE_COLS, NORMALIZED_COLS
from .heroes_loader import MULTIPLE_ATTRS
import torch
import math
from sklearn import preprocessing
import pandas as pd
from .util import sig_to_tanh_range

HERO_COLS=["id", "lane", "roles", "specialities"]
TARGET_COLS=["left_victory", "scores_sum_diff_norm", "match_duration_norm"]

class SymmetricScaler:
    def __init__(self):
        pass

    def fit(self, X):
        mean = X.mean(axis=0)
        minmax = X.min(axis=0), X.max(axis=0)
        diff = [abs(m-mean) for m in minmax]
        max_diff = np.maximum(*diff)
        self.mean = mean
        self.max_diff = max_diff
        return self

    def transform(self, X):
        X_std = (X - self.mean) / self.max_diff
        return X_std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_std):
        X = X_std * self.max_diff + self.mean
        return X

def normalize(df):
    scaler = SymmetricScaler().fit(df[NORMALIZE_COLS])
    df[NORMALIZED_COLS] = scaler.transform(df[NORMALIZE_COLS])
    return scaler

def get_mixed(df_heroes, x):
    return [a for i in range(2) for a in df_heroes[f"{x}_{i}"].tolist()]
def get_unique(mixed):
    return sorted(list(set(mixed)), key=lambda x: (x is None, x))
def get_basic_c(c):
    return c.split("_", maxsplit=1)[0]

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
    
def create_embedding(n):
    return torch.nn.Embedding(n, math.ceil(math.sqrt(n)))

ATTR_CLASSES = {
    "id": 120,
    "lane": 5,
    "roles": 7,
    "specialities": 16,
}
def create_embedding_sizes(
    columns, 
    f=lambda x: int(2*math.ceil(math.sqrt(x)))
):
    if isinstance(columns[0], int):
        classes = columns
    else:
        classes = [ATTR_CLASSES[get_basic_c(c)] for c in columns]
    return [(c, f(c)) for c in classes]


class HeroEmbedder(torch.nn.Module):
    def __init__(self, sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embeddings = []
        for s in sizes:
            if isinstance(s, int):
                embedding = embeddings[s]
            else:
                embedding = torch.nn.Embedding(*s)
            embeddings.append(embedding)
        embeddings = torch.nn.ModuleList(embeddings)
        self.embeddings = embeddings
        self.main_dim = embeddings[0].weight.shape[-1]
        self.dim = sum(e.weight.shape[-1] for e in embeddings)

    def embed_batch(self, encoded_tensor):
        split_encoded = torch.split(encoded_tensor, 1, dim=-1)
        split_encoded = [torch.squeeze(e, dim=-1) for e in split_encoded]
        #print(len(split_encoded), split_encoded[0].shape)
        split_embed = [
            self.embeddings[i](split_encoded[i]) 
            for i in range(len(split_encoded))
        ]
        embedded = torch.cat(split_embed, dim=-1)
        #print(split_embed.shape)
        return embedded

    def forward(self, encoded_tensor):
        return self.embed_batch(encoded_tensor)

    def __call__(self, encoded_tensor):
        return self.embed_batch(encoded_tensor)
    
    def reverse(self, sample):
        sample = sample[..., :self.main_dim]
        distance = torch.norm(self.embeddings[0].weight.data - sample, dim=1)
        nearest = torch.argmin(distance)
        return nearest
    
def calc_objective(target):
    target["objective"] = target["left_victory"] + (target["scores_sum_diff_norm"] / (2 + target["match_duration_norm"]))
    return target["objective"]

def extract_target(df):
    return torch.Tensor(df[TARGET_COLS].to_numpy().astype(float))

def split_dataframe(df, points, rand=42):
    return np.split(
        df.sample(frac=1, random_state=rand), 
        [int(x*len(df)) for x in points]
    )
