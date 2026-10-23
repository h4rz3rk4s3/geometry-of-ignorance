"""Baseline comparison for the non-knowledge revision (meta-review point 2).

Labels are 0/1 (1 = NK).

python run.py \
    --train ../datasets/knowledge_non_knowledge_v3_training_rogue_filtered_both_models_with_knowledge_neutral_oversampled.csv \
    --eval Synth=../datasets/knowledge_non_knowledge_v3_evaluation_rogue_filtered_both_models_with_knowledge_neutral_oversampled.csv \
    --eval GT=../datasets/knowledge_non_knowledge_v3_categorized.csv \
    --eval OoD=../datasets/OpenVox_non_knowledge_eval.csv \
    --eval Ind=../datasets/independent_evaluation_set.csv \
    --model RoBERTa-NK-Large=/Users/giulianowietig/PycharmProjects/models/roberta-non-knowledge-v8-large \
    --model RoBERTa-NK-Base=/Users/giulianowietig/PycharmProjects/models/roberta-non-knowledge-v8-base \
    --model ModernBERT-NK=/Users/giulianowietig/PycharmProjects/models/ModernBERT-non-knowledge-v8 \
    --mask --out results/

"""
import argparse
from pathlib import Path

import pandas as pd

from baselines import CueMatcher, TfidfLR
from metrics import SUBSETS, add_cue_columns, cue_stats, evaluate

def to_binary(label, is_training: bool = False) -> int:
    label = str(label).strip()
    # OoD uses 0 = K/N, 2 = NK (no 1); GT and v8 use 0 = K/N, 1 = NK
    mapping = {"0": 0, "1": 1} if is_training else {"0": 0, "1":0, "2": 1}
    if label not in mapping:
        raise ValueError(f"Unknown label {label!r} (is_training={is_training})")
    return mapping[label]


def load(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    is_training = any(k in path for k in ("training", "independent", "evaluation"))
    print(path)
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "statement", label_col: "final_label"})[["statement", "final_label"]]
    df["statement"] = df["statement"].fillna("").astype(str)
    df["final_label"] = df["final_label"].apply(to_binary, is_training=is_training) # add is_odd to to_binary call
    return df.reset_index(drop=True)


def name_path(arg: str) -> tuple[str, str]:
    name, path = arg.split("=", 1)
    return name, path


def markdown_table(results: pd.DataFrame, metric: str, datasets: list[str]) -> str:
    """Rows = models, columns = dataset x subset; header row gives n (n_NK)."""
    cols = [(d, s) for d in datasets for s in SUBSETS]
    header = "| Model | " + " | ".join(f"{d} {s}" for d, s in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    first = results.drop_duplicates(["dataset", "subset"]).set_index(["dataset", "subset"])
    n_row = "| *n (n_NK)* | " + " | ".join(
        f"{first.loc[c, 'n']} ({first.loc[c, 'n_nk']})" for c in cols) + " |"
    lines = [f"**{metric}**", "", header, sep, n_row]
    for model in results["model"].unique():
        r = results[results["model"] == model].set_index(["dataset", "subset"])
        cells = ["–" if pd.isna(r.loc[c, metric]) else f"{r.loc[c, metric]:.2f}" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", help="synthetic training CSV, needed for TF-IDF + LR")
    p.add_argument("--eval", action="append", required=True, type=name_path,
                   help="NAME=path.csv, repeatable (e.g. GT=..., OoD=...)")
    p.add_argument("--model", action="append", default=[], type=name_path,
                   help="NAME=path to a fine-tuned HF model, repeatable")
    p.add_argument("--mask", action="store_true", help="to run cue-masked evaluation")
    p.add_argument("--nk-id", type=int, default=None, help="NK class index if not in the model config")
    p.add_argument("--text-col", default="statement")
    p.add_argument("--label-col", default="final_label")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {name: add_cue_columns(load(path, args.text_col, args.label_col))
                for name, path in args.eval}

    # Collect (row name, predict function) pairs
    systems = [(CueMatcher.name, CueMatcher().predict)]
    if args.train:
        train = load(args.train, args.text_col, args.label_col)
        tfidf = TfidfLR().fit(train["statement"].tolist(), train["final_label"].tolist())
        systems.append((tfidf.name, tfidf.predict))
        if args.mask:
            systems.append((f"{tfidf.name} (cue-masked)", lambda t: tfidf.predict(t, mask=True)))
    if args.model:
        from encoders import Encoder  # import torch only when needed
        for name, path in args.model:
            enc = Encoder(name, path, nk_id=args.nk_id)
            systems.append((name, enc.predict))
            if args.mask:
                systems.append((f"{name} (cue-masked)", lambda t, e=enc: e.predict(t, mask=True)))

    results, stats = [], []
    for ds_name, df in datasets.items():
        texts = df["statement"].tolist()
        for sys_name, predict in systems:
            print(f"Predicting {sys_name} on {ds_name} ...")
            df[sys_name] = predict(texts)
            results += evaluate(df, sys_name, sys_name, ds_name)
        df.to_csv(out / f"predictions_{ds_name}.csv", index=False)  # error analysis / CIs, per example
        stats.append(cue_stats(df, ds_name))

    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    pd.concat(stats).to_csv(out / "cue_stats.csv", index=False)

    tables = [markdown_table(results, m, list(datasets)) for m in ["f1_nk", "f1_macro", "f1_weighted", "accuracy"]]
    (out / "table.md").write_text("\n\n".join(tables) + "\n")
    print("\n" + "\n\n".join(tables))
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
