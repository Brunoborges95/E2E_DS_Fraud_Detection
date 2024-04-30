import numpy as np
import pandas as pd
import sklearn.metrics as m
from sklearn.preprocessing import LabelEncoder

def pre_cat(df):
    df = df.assign(g = lambda x: x.g.fillna('UNKNOW').astype('category'),
                    o = lambda x: x.o.fillna('UNKNOW').astype('category'),
                    p = lambda x: x.p.fillna('UNKNOW').astype('category'),
                    )
    return df

def pre_xgb(df):
    df = df.assign(g = lambda x: x.g.astype('category'),
                    o = lambda x: x.o.astype('category'),
                    p = lambda x: x.p.astype('category'),
                    )
    return df

def encode_categorical_to_integers(df):
    label_encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == "category":
            df[column] = label_encoder.fit_transform(df[column])
    return df.fillna(-1)


def evaluate(y_true, y_hat):
    roc_auc_score = m.roc_auc_score(y_true, y_hat)
    brier_score = m.brier_score_loss(y_true, y_hat)
    logloss = m.log_loss(y_true, y_hat)
    f5s = []
    cutoffs = np.linspace(0, 1, 20)
    for cutoff in cutoffs:
        predictions = (y_hat > cutoff).astype(int)
        f5s.append(m.fbeta_score(y_true, predictions, beta=5))

    return {'roc_auc_score': roc_auc_score,
                    'brier_score': brier_score,
                    'logloss': logloss,
                    'f5s': max(f5s)}