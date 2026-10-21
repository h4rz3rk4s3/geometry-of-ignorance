"""Metrics per subset (all / +cue / -cue) and per-cue firing statistics."""
import math

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cues import CUES, find_cues

SUBSETS = ["all", "+cue", "-cue"]


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all":
        return pd.Series(True, index=df.index)
    return df["has_cue"] if subset == "+cue" else ~df["has_cue"]


def scores(y_true, y_pred) -> dict:
    """NK-F1, macro-F1, weighted-F1, accuracy plus counts.

    If a subset contains only one class (e.g. no NK example without a cue), the F1
    variants are undefined and reported as NaN; accuracy is still meaningful
    (for a K/N-only subset it is 1 - false-positive rate).
    """
    n, n_nk = len(y_true), int(sum(y_true))
    nan = math.nan
    if n == 0:
        return {"n": 0, "n_nk": 0, "f1_nk": nan, "f1_macro": nan, "f1_weighted": nan, "accuracy": nan}
    both = 0 < n_nk < n
    return {
        "n": n,
        "n_nk": n_nk,
        "f1_nk": f1_score(y_true, y_pred, pos_label=1, zero_division=0) if n_nk else nan,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if both else nan,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) if both else nan,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate(df: pd.DataFrame, pred_col: str, model: str, dataset: str) -> list[dict]:
    rows = []
    for subset in SUBSETS:
        part = df[subset_mask(df, subset)]
        rows.append({"model": model, "dataset": dataset, "subset": subset,
                     **scores(part["final_label"].tolist(), part[pred_col].tolist())})
    return rows


def cue_stats(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """For each cue: how often it fires on K/N vs. NK examples, and its precision for NK."""
    rows = []
    for cue in CUES:
        fires = df["cues"].apply(lambda found: cue in found)
        on_nk = int((fires & (df["final_label"] == 1)).sum())
        on_k = int((fires & (df["final_label"] == 0)).sum())
        rows.append({"dataset": dataset, "cue": cue, "fires_on_nk": on_nk, "fires_on_k": on_k,
                     "precision_nk": on_nk / (on_nk + on_k) if on_nk + on_k else math.nan})
    return pd.DataFrame(rows)


def add_cue_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cues"] = df["statement"].apply(find_cues)
    df["has_cue"] = df["cues"].apply(bool)
    return df
