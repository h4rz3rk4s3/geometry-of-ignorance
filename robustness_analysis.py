#!/usr/bin/env python3
"""
Robustness analyses for a binary Knowledge (0) / Non-Knowledge (1) classifier
evaluated on a small OoD set -- no retraining of the model, no new labels.

Steps (select with --steps, default: all that the inputs allow)
  shift       Train/test leakage via near-duplicates, distance of every item to the
              training data, performance stratified by that distance, a check whether
              the observed OoD accuracy matches what the test set predicts for data
              that far away, and a domain-classifier AUC (how separable is each set
              from the training data).
  baselines   Lexicon, TF-IDF + logistic regression and (optional) zero-shot LLM
              baselines on the same items; paired exact McNemar tests with Holm
              correction and a bootstrap CI for "does our model drop more from test to
              OoD than the baseline?".
  invariance  CheckList-style invariance (INV) and directional (DIR) tests
              (Ribeiro et al., 2020).
  errors      Exports every OoD error with context for manual coding;
              --coded-errors summarises a coded file afterwards.
  atc         Average Threshold Confidence (Garg et al., 2022), validated on the
              labelled OoD set and applied to --unlabeled if given.

Model outputs, embeddings and LLM answers are cached in <outdir>/cache, so re-runs
(e.g. after adding --unlabeled or --coded-errors) do not recompute them.

Example
  python robustness_analysis.py \
      --model /path/to/models/roberta-non-knowledge-v8-base \
      --train datasets/train.csv --test datasets/test.csv \
      --ood datasets/OpenVox_non_knowledge_eval.csv --label-map ood:1=0,2=1 \
      --llm-model qwen3-32b --llm-guideline guideline.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_uncertainty as f1u  # noqa: E402

log = logging.getLogger("robustness")
ALL_STEPS = ["shift", "baselines", "invariance", "errors", "atc"]
MAIN_METRICS = ["f1_pos", "f1_macro", "balanced_acc", "mcc"]


# ----------------------------------------------------------------------------- utils
def text_hash(texts, *extra) -> str:
    h = hashlib.sha1()
    for e in extra:
        h.update(str(e).encode()); h.update(b"\0")
    for t in texts:
        h.update(str(t).encode()); h.update(b"\0")
    return h.hexdigest()[:12]


def fmt(v, lo=None, hi=None, d=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    s = f"{v:.{d}f}"
    if lo is not None and not np.isnan(lo):
        s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return s


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def cp_interval(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    ci = stats.binomtest(int(k), int(n)).proportion_ci(1 - alpha, method="exact")
    return ci.low, ci.high


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adj); out[order] = adj
    return out


# --------------------------------------------------------------------------- metrics
def point(y, yhat) -> dict:
    return {k: float(v) for k, v in f1u.metrics_from_counts(*f1u.counts(y, yhat)).items()}


def defined(y, metric) -> bool:
    npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
    return npos > 0 if metric == "f1_pos" else (npos > 0 and nneg > 0)


def fast_jackknife(y, yhat):
    """Leave-one-out only removes one confusion cell -> 4 distinct values."""
    cell = np.where(y == 1, np.where(yhat == 1, 0, 2), np.where(yhat == 1, 1, 3))
    base = np.array(f1u.counts(y, yhat), float)
    loo = base[None, :] - np.eye(4)
    m = f1u.metrics_from_counts(*loo.T)
    return {k: np.asarray(v)[cell] for k, v in m.items()}


def metric_ci(y, yhat, metric, B, seed=0, alpha=0.05):
    """Point estimate + stratified BCa bootstrap CI (see f1_uncertainty.py)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    if len(y) == 0 or not defined(y, metric):
        return np.nan, np.nan, np.nan
    obs = point(y, yhat)[metric]
    boot = f1u.stratified_bootstrap(y, yhat, B, np.random.default_rng(seed))[metric]
    lo, hi = f1u.bca_interval(obs, boot, fast_jackknife(y, yhat)[metric], alpha)
    return obs, float(lo), float(hi)


def strat_indices(y, B, rng):
    parts = [rng.choice(np.flatnonzero(y == c), (B, int(np.sum(y == c))))
             for c in (0, 1) if np.any(y == c)]
    return np.hstack(parts)


def metric_from_idx(y, yhat, idx, metric):
    yt, yp = y[idx], yhat[idx]
    tp = np.sum((yt == 1) & (yp == 1), 1); fp = np.sum((yt == 0) & (yp == 1), 1)
    fn = np.sum((yt == 1) & (yp == 0), 1); tn = np.sum((yt == 0) & (yp == 0), 1)
    return f1u.metrics_from_counts(tp, fp, fn, tn)[metric]


def drop_difference_ci(y_te, pm_te, pb_te, y_ood, pm_ood, pb_ood, metric, B, rng, chunk=500):
    """95% percentile CI of (drop_model - drop_baseline), drop = test - OoD.
    Model and baseline share the same resamples (paired within each set)."""
    out = []
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        it, io = strat_indices(y_te, b, rng), strat_indices(y_ood, b, rng)
        dm = metric_from_idx(y_te, pm_te, it, metric) - metric_from_idx(y_ood, pm_ood, io, metric)
        db = metric_from_idx(y_te, pb_te, it, metric) - metric_from_idx(y_ood, pb_ood, io, metric)
        out.append(dm - db)
    return np.quantile(np.concatenate(out), [0.025, 0.975])


def mcnemar_exact(y, a, b):
    """Exact McNemar on paired correctness. n01: a right & b wrong; n10: reverse."""
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


# ------------------------------------------------------------------------------ data
def parse_label_maps(specs):
    maps = {}
    for spec in specs or []:
        name, pairs = spec.split(":", 1)
        maps[name] = {int(a): int(b) for a, b in (p.split("=") for p in pairs.split(","))}
    return maps


def pick_text_col(df, requested):
    if requested:
        return requested
    # same precedence as the evaluation script: rephrased_statement is renamed to statement
    for c in ("rephrased_statement", "statement", "text"):
        if c in df.columns:
            return c
    raise ValueError(f"no text column found in {list(df.columns)}; use --text-col")


def load_set(path, name, args, maps, labelled=True, strict=True) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = pick_text_col(df, args.text_col)
    df = df[df[col].notna()].reset_index(drop=True)
    out = pd.DataFrame({"text": df[col].astype(str).values})
    log.info("%s: %d rows from %s (text column '%s')", name, len(out), path, col)
    if labelled:
        y = pd.to_numeric(df[args.label_col], errors="coerce")
        if name in maps:
            y = y.map(lambda v: maps[name].get(int(v), v) if pd.notna(v) else v)
        ok = y.isin([0, 1]).values
        if not ok.all():
            bad = sorted(pd.Series(y[~ok]).dropna().unique().tolist())
            if strict:
                raise ValueError(f"{name}: label values {bad} (or missing) are not 0/1 after mapping; "
                                 f"use --label-map {name}:FROM=TO,...")
            log.warning("%s: dropping %d rows with labels %s", name, int((~ok).sum()), bad)
        out = out[ok].reset_index(drop=True)
        out["y"] = y[ok].astype(int).values
        log.info("%s: %d Knowledge / %d Non-Knowledge", name, int((out.y == 0).sum()), int((out.y == 1).sum()))
    return out


# ----------------------------------------------------------------------------- model
class HFClassifier:
    def __init__(self, path, device, dtype, logit_indices, batch_size, max_length):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        n = self.model.config.num_labels
        self.idx = list(logit_indices)
        if len(self.idx) != 2 or max(self.idx) >= n:
            raise ValueError(f"model has {n} output labels; --logit-indices must name two (Knowledge,Non-Knowledge)")
        if n > 2:
            log.warning("model has %d labels; using logits %s as (Knowledge, Non-Knowledge)", n, self.idx)
        self.bs, self.ml = batch_size, max_length
        log.info("classifier loaded on %s (%s)", device, dtype)

    def _forward(self, texts, hidden):
        logits, embs = [], []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.bs):
                enc = self.tok(list(texts[i:i + self.bs]), truncation=True, max_length=self.ml,
                               padding=True, return_tensors="pt").to(self.device)
                out = self.model(**enc, output_hidden_states=hidden)
                logits.append(out.logits.float().cpu().numpy()[:, self.idx])
                if hidden:  # mean-pooled last hidden state
                    h = out.hidden_states[-1].float()
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        return np.vstack(logits), (np.vstack(embs) if hidden else None)

    def logits(self, texts):
        return self._forward(texts, False)[0]

    def embed(self, texts):
        return self._forward(texts, True)[1]


class Ctx:
    """Holds data, lazily loaded models, caches and the report."""

    def __init__(self, args, classifier=None):
        self.args = args
        self.out = Path(args.outdir); self.out.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.out / "cache"; self.cache_dir.mkdir(exist_ok=True)
        self.sets: dict[str, pd.DataFrame] = {}
        self.report: list[str] = []
        self._clf = classifier
        self._embedder = None
        self._emb, self._nn = {}, {}
        self.baseline_preds: dict[str, dict[str, np.ndarray]] = {}

    def section(self, title, *parts):
        self.report.append(f"\n## {title}\n")
        self.report.extend(parts)

    @property
    def clf(self):
        if self._clf is None:
            a = self.args
            self._clf = HFClassifier(a.model, a.device, a.dtype, a.logit_indices, a.batch_size, a.max_length)
        return self._clf

    def _cached(self, kind, texts, extra, fn):
        p = self.cache_dir / f"{kind}_{text_hash(texts, *extra)}.npy"
        if p.exists():
            return np.load(p)
        arr = np.asarray(fn(list(texts)), dtype=np.float32)
        np.save(p, arr)
        return arr

    def logits(self, texts):
        a = self.args
        if len(texts) == 0:
            return np.zeros((0, 2), np.float32)
        return self._cached("logits", texts, (a.model, a.dtype, a.logit_indices, a.max_length),
                            lambda t: self.clf.logits(t))

    def predict_all(self):
        for name, df in self.sets.items():
            if name == "train":
                continue
            z = self.logits(df.text.tolist())
            m = z[:, 1] - z[:, 0]
            df["margin"], df["p_nk"], df["pred"] = m, expit(m), (m > 0).astype(int)

    # -- embeddings / nearest neighbours
    def embeddings(self, name):
        if name in self._emb:
            return self._emb[name]
        a = self.args
        texts = self.sets[name].text.tolist()
        if a.embedder == "tfidf":
            fn, extra = self._tfidf_embedder(), ("tfidf", text_hash(self.sets["train"].text))
        elif a.embedder == "classifier":
            fn, extra = (lambda t: self.clf.embed(t)), ("clf", a.model, a.dtype, a.max_length)
        else:
            fn, extra = self._st_embedder(), ("st", a.embedder)
        E = self._cached("emb", texts, extra, fn)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
        self._emb[name] = E.astype(np.float32)
        return self._emb[name]

    def _st_embedder(self):
        def fn(texts):
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                dev = None if self.args.device == "auto" else self.args.device
                self._embedder = SentenceTransformer(self.args.embedder, device=dev)
            return self._embedder.encode(texts, batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=False)
        return fn

    def _tfidf_embedder(self):
        def fn(texts):
            if self._embedder is None:
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                X = vec.fit_transform(self.sets["train"].text.tolist())
                svd = TruncatedSVD(min(300, X.shape[1] - 1), random_state=0).fit(X)
                self._embedder = (vec, svd)
            vec, svd = self._embedder
            return svd.transform(vec.transform(texts))
        return fn

    def nn(self, name):
        if name not in self._nn:
            self._nn[name] = nn_search(self.embeddings(name), self.embeddings("train"))
        return self._nn[name]


def nn_search(Q, R, chunk=256):
    sims, idx = np.empty(len(Q), np.float32), np.empty(len(Q), int)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        j = S.argmax(1)
        idx[i:i + chunk], sims[i:i + chunk] = j, S[np.arange(len(S)), j]
    return sims, idx


# ------------------------------------------------------------------------- lexicon
# Generic English uncertainty / ignorance markers. Placeholder: replace it with your
# own marker inventory via --lexicon (one regex per line, '#' comments allowed).
DEFAULT_LEXICON = [
    r"\bnot (?:really |entirely |completely |quite |100% )?(?:sure|certain|clear)\b",
    r"\bunsure\b", r"\buncertain(?:ty)?\b", r"\bunclear\b", r"\bunknown\b",
    r"\bno (?:idea|clue)\b", r"\bidk\b", r"\b(?:nobody|no one) (?:knows|understands)\b",
    r"\b(?:do|does|did)(?: not|n't|n’t|nt) (?:know|understand|get why)\b",
    r"\bwonder(?:ing)?\b", r"\bnot (?:yet )?(?:known|understood|documented|explained)\b",
    r"\b(?:have|has|need|needs) to (?:investigate|check|verify|figure out|find out|look into)\b",
    r"\bfigur(?:e|ing) out\b", r"\binvestigat\w*\b", r"\blook(?:ing)? into\b",
    r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b", r"\bmight\b",
    r"\b(?:i|we) (?:think|guess|suspect|assume|believe)\b", r"\bpresumably\b", r"\bapparently\b",
    r"\bseems? (?:to|like)\b", r"\bconfus(?:ed|ing|ion)\b", r"\bmyster\w*\b",
    r"\b(?:any|anyone|anybody|someone|somebody) (?:ideas?|know|knows)\b",
    r"\bcan (?:anyone|somebody|someone)\b", r"\bTBD\b", r"\bTODO\b", r"\?",
]


class Lexicon:
    def __init__(self, path=None):
        if path:
            pats = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.lstrip().startswith("#")]
        else:
            pats = DEFAULT_LEXICON
        self.rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I)

    def hits(self, text):
        return [m.group(0) for m in self.rx.finditer(text)]

    def spans(self, text):
        return [(m.start(), m.end()) for m in self.rx.finditer(text)]

    def predict(self, texts):
        return np.array([int(bool(self.rx.search(t))) for t in texts])

    def remove(self, text):
        s = self.rx.sub(lambda m: "." if m.group(0) == "?" else " ", text)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([,.;:])\1+", r"\1", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^[\s,;:.\-]+", "", s).strip()
        if not s or s == text.strip():
            return None
        return s[0].upper() + s[1:]


# ----------------------------------------------------------------------- LLM client
DEFAULT_PROMPT = """You are annotating statements from software-engineering communication.

Annotation guideline:
{guideline}

Classify the statement as KNOWLEDGE or NON-KNOWLEDGE according to the guideline.
Answer with exactly one label and nothing else."""


def parse_llm(raw):
    s = re.sub(r"<think>.*?</think>", " ", raw or "", flags=re.S | re.I).upper()
    s = re.sub(r"[-_]", " ", s)
    if re.search(r"\bNON\s*KNOWLEDGE\b", s):
        return 1
    if re.search(r"\bKNOWLEDGE\b", s):
        return 0
    return -1


class LLMClient:
    """OpenAI-compatible /chat/completions (llama.cpp server, Ollama, LM Studio, vLLM, ...)."""

    def __init__(self, a, cache_path):
        import os
        import requests
        self.a, self.http = a, requests.Session()
        guideline = Path(a.llm_guideline).read_text(encoding="utf-8")
        template = Path(a.llm_prompt).read_text(encoding="utf-8") if a.llm_prompt else DEFAULT_PROMPT
        self.system = template.replace("{guideline}", guideline)
        self.url = a.llm_base_url.rstrip("/") + "/chat/completions"
        key = os.environ.get("LLM_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.prefix = text_hash([self.system], a.llm_model, a.llm_max_tokens)
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.prompt_hash = self.prefix

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache))

    def raw(self, text):
        k = self.prefix + ":" + hashlib.sha1(text.encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        body = {"model": self.a.llm_model, "temperature": 0, "max_tokens": self.a.llm_max_tokens,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": f"Statement: {text}"}]}
        for attempt in range(4):
            try:
                r = self.http.post(self.url, json=body, headers=self.headers, timeout=self.a.llm_timeout)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"].get("content") or ""
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    self.save()
                    raise RuntimeError(f"LLM request failed: {e}") from e
                time.sleep(2 ** attempt)
        self.cache[k] = out
        if len(self.cache) % 25 == 0:
            self.save()
        return out

    def classify(self, texts):
        preds = []
        for i, t in enumerate(texts):
            preds.append(parse_llm(self.raw(t)))
            if (i + 1) % 50 == 0:
                log.info("  LLM: %d/%d", i + 1, len(texts))
        self.save()
        return np.array(preds)


# ============================================================================ steps
def run_overview(ctx):
    a, rows = ctx.args, []
    for name in ("test", "ood"):
        df = ctx.sets[name]
        row = {"set": name, "n": len(df), "Knowledge/Non-Knowledge": f"{(df.y == 0).sum()}/{(df.y == 1).sum()}"}
        for m in MAIN_METRICS:
            row[m] = fmt(*metric_ci(df.y.values, df.pred.values, m, a.B, a.seed))
        rows.append(row)
    ctx.section("Model performance (stratified BCa 95% CIs)", md_table(pd.DataFrame(rows)),
                "Check that these point estimates reproduce your evaluation script. If they do not, "
                "the label mapping, text column or dtype differs.\n")


# --------------------------------------------------------------------------- shift
def char_nn(ctx, names):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          max_features=300_000, sublinear_tf=True)
    R = vec.fit_transform(ctx.sets["train"].text.tolist())
    out = {}
    for n in names:
        Q = vec.transform(ctx.sets[n].text.tolist())
        sims, idx = np.empty(Q.shape[0]), np.empty(Q.shape[0], int)
        for i in range(0, Q.shape[0], 256):
            S = (Q[i:i + 256] @ R.T).toarray()
            j = S.argmax(1)
            idx[i:i + 256], sims[i:i + 256] = j, S[np.arange(len(S)), j]
        out[n] = (sims, idx)
    return out


def expected_accuracy(sim_te, correct_te, sim_ood, B, rng):
    """Fit P(correct | nn-similarity) on the test set, predict OoD accuracy.
    Returns predictive intervals with parameter (bootstrap) and sampling (Bernoulli)
    uncertainty, plus the same for a distance-agnostic reference."""
    from sklearn.linear_model import LogisticRegression
    c = correct_te.astype(int)
    if c.min() == c.max():
        return None
    Xte, Xo = sim_te.reshape(-1, 1), sim_ood.reshape(-1, 1)
    fit = lambda X, y: LogisticRegression(C=1e6, max_iter=2000).fit(X, y)  # noqa: E731
    base = fit(Xte, c)
    acc_dist, acc_const, slopes = np.empty(B), np.empty(B), np.empty(B)
    n = len(c)
    for b in range(B):
        i = rng.integers(0, n, n)
        cb = c[i]
        if cb.min() == cb.max():
            p, slopes[b] = np.full(len(Xo), cb.mean()), 0.0
        else:
            m = fit(Xte[i], cb)
            p, slopes[b] = m.predict_proba(Xo)[:, 1], m.coef_[0, 0]
        acc_dist[b] = rng.binomial(1, p).mean()
        acc_const[b] = rng.binomial(len(Xo), cb.mean()) / len(Xo)
    return dict(slope=base.coef_[0, 0], slope_ci=np.quantile(slopes, [0.025, 0.975]),
                expected=base.predict_proba(Xo)[:, 1].mean(),
                pi=np.quantile(acc_dist, [0.025, 0.975]), sims=acc_dist,
                const_expected=c.mean(), const_pi=np.quantile(acc_const, [0.025, 0.975]),
                const_sims=acc_const)


def domain_auc(E_ref, E_x, n_x, reps, seed):
    """Cross-validated AUC of a logistic probe separating train from set X.
    X is subsampled to n_x items per repetition so sets of different size are comparable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rng, aucs = np.random.default_rng(seed), []
    for r in range(reps):
        xs = E_x[rng.choice(len(E_x), min(n_x, len(E_x)), replace=False)]
        ref = E_ref[rng.choice(len(E_ref), min(len(E_ref), max(10 * len(xs), 200)), replace=False)]
        X = np.vstack([ref, xs]); y = np.r_[np.zeros(len(ref)), np.ones(len(xs))]
        cv = StratifiedKFold(5, shuffle=True, random_state=r)
        p = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=3000), X, y,
                              cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))


def run_shift(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = ctx.args
    tr, te, od = ctx.sets["train"], ctx.sets["test"], ctx.sets["ood"]
    rng = np.random.default_rng(a.seed)
    parts = []

    # 1) near-duplicates / leakage ------------------------------------------------
    cnn = char_nn(ctx, ["test", "ood"])
    dup_rows = []
    for name in ("test", "ood"):
        df = ctx.sets[name]
        es, ei = ctx.nn(name)
        cs, ci = cnn[name]
        df["nn_sim"], df["nn_idx"], df["char_sim"] = es, ei, cs
        dup = (es >= a.dup_emb) | (cs >= a.dup_char)
        df["near_dup"] = dup
        nb = np.where(cs >= a.dup_char, ci, ei)  # neighbour that triggered the flag
        exp = pd.DataFrame({"text": df.text, "label": df.y, "pred": df.pred, "emb_sim": es.round(3),
                            "char_sim": cs.round(3), "nearest_train_text": tr.text.values[nb],
                            "nearest_train_label": tr.y.values[nb]})[dup]
        exp.sort_values(["emb_sim", "char_sim"], ascending=False).to_csv(
            ctx.out / f"near_duplicates_{name}.csv", index=False)
        agree = float(np.mean(exp.label == exp.nearest_train_label)) if len(exp) else np.nan
        dup_rows.append({"set": name, "n": len(df), "near-duplicates of train": f"{dup.sum()} ({dup.mean():.1%})",
                         "same label as train neighbour": fmt(agree, d=2)})
    te_all = [metric_ci(te.y.values, te.pred.values, m, a.B, a.seed) for m in ("f1_pos", "balanced_acc")]
    keep = ~te.near_dup.values
    te_dd = [metric_ci(te.y.values[keep], te.pred.values[keep], m, a.B, a.seed) for m in ("f1_pos", "balanced_acc")]
    dd = pd.DataFrame([{"test subset": "all", "n": len(te), "f1_pos": fmt(*te_all[0]), "balanced_acc": fmt(*te_all[1])},
                       {"test subset": "without near-duplicates", "n": int(keep.sum()),
                        "f1_pos": fmt(*te_dd[0]), "balanced_acc": fmt(*te_dd[1])}])
    parts += ["### Near-duplicates between evaluation sets and training data\n",
              f"Flagged if embedding cosine >= {a.dup_emb} or character 3-5-gram TF-IDF cosine >= {a.dup_char} "
              f"to the nearest training item (embedder: `{a.embedder}`). Thresholds are heuristics: read the pairs in "
              "`near_duplicates_test.csv` before reporting numbers.\n",
              md_table(pd.DataFrame(dup_rows)), md_table(dd),
              "If the de-duplicated test score is clearly lower, part of the in-distribution score is memorisation "
              "and the de-duplicated number is the fairer reference for the OoD comparison.\n"]

    # 2) how far is each set from the training data? -------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(min(te.nn_sim.min(), od.nn_sim.min()), 1, 40)
    ax.hist(te.nn_sim, bins=bins, density=True, alpha=0.5, label="test")
    ax.hist(od.nn_sim, bins=bins, density=True, alpha=0.5, label="OoD")
    ax.set_xlabel("cosine similarity to nearest training item"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); fig.savefig(ctx.out / "nn_similarity_hist.png", dpi=200); plt.close(fig)

    auc_te = domain_auc(ctx.embeddings("train"), ctx.embeddings("test"), len(od), a.auc_reps, a.seed)
    auc_od = domain_auc(ctx.embeddings("train"), ctx.embeddings("ood"), len(od), a.auc_reps, a.seed)
    shift_tab = pd.DataFrame([
        {"set": "test", "median nn-sim": f"{te.nn_sim.median():.3f}",
         "IQR": f"{te.nn_sim.quantile(.25):.3f}-{te.nn_sim.quantile(.75):.3f}",
         "domain AUC vs train (mean, range)": f"{auc_te[0]:.3f} ({auc_te[1]:.3f}-{auc_te[2]:.3f})"},
        {"set": "OoD", "median nn-sim": f"{od.nn_sim.median():.3f}",
         "IQR": f"{od.nn_sim.quantile(.25):.3f}-{od.nn_sim.quantile(.75):.3f}",
         "domain AUC vs train (mean, range)": f"{auc_od[0]:.3f} ({auc_od[1]:.3f}-{auc_od[2]:.3f})"}])
    parts += ["### Magnitude of the shift\n",
              f"Domain AUC: cross-validated logistic probe on embeddings separating training items from the set "
              f"(both subsampled to n={len(od)}, {a.auc_reps} repetitions). ~0.5 = indistinguishable from training "
              "data, ~1.0 = trivially separable. The test set should be near 0.5; if it is not, the split is not i.i.d.\n",
              md_table(shift_tab), "![](nn_similarity_hist.png)\n"]

    # 3) performance stratified by distance ----------------------------------------
    te["dist_bin"] = pd.qcut(te.nn_sim, a.n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in te.groupby("dist_bin"):
        rows.append(dict(group=f"test bin {int(b) + 1}", n=len(g), n_pos=int(g.y.sum()),
                         sim_med=g.nn_sim.median(), sim_lo=g.nn_sim.min(), sim_hi=g.nn_sim.max(),
                         **{m: metric_ci(g.y.values, g.pred.values, m, a.B, a.seed) for m in ("f1_pos", "balanced_acc")}))
    rows.append(dict(group="OoD", n=len(od), n_pos=int(od.y.sum()), sim_med=od.nn_sim.median(),
                     sim_lo=od.nn_sim.min(), sim_hi=od.nn_sim.max(),
                     **{m: metric_ci(od.y.values, od.pred.values, m, a.B, a.seed) for m in ("f1_pos", "balanced_acc")}))
    strat = pd.DataFrame(rows)
    strat_out = strat.copy()
    for m in ("f1_pos", "balanced_acc"):
        strat_out[m] = [fmt(*v) for v in strat[m]]
    strat_out["similarity range"] = [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(strat.sim_lo, strat.sim_hi)]
    strat_out = strat_out[["group", "n", "n_pos", "similarity range", "f1_pos", "balanced_acc"]]
    strat_out.to_csv(ctx.out / "performance_by_distance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, m, lab in zip(axes, ("f1_pos", "balanced_acc"), ("F1 (Non-Knowledge)", "Balanced accuracy")):
        for is_ood, style in ((False, dict(fmt="o-", color="C0", label="test (distance bins)")),
                              (True, dict(fmt="s", color="C3", ms=8, label="OoD"))):
            sub = strat[(strat.group == "OoD") == is_ood]
            v = np.array([x[0] for x in sub[m]]); lo = np.array([x[1] for x in sub[m]]); hi = np.array([x[2] for x in sub[m]])
            ax.errorbar(sub.sim_med, v, yerr=[v - lo, hi - v], capsize=3, **style)
        ax.set_xlabel("median cosine similarity to nearest training item"); ax.set_ylabel(lab)
    axes[0].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ctx.out / "performance_by_distance.png", dpi=200); plt.close(fig)
    parts += ["### Performance as a function of distance to the training data\n",
              f"Test set split into {te.dist_bin.nunique()} quantile bins of nearest-neighbour similarity; OoD shown "
              "as a separate point. If the OoD point lies on the trend of the test bins, the OoD drop is explained by "
              "distance from the training data. If it lies clearly below, something else about the OoD data matters.\n",
              md_table(strat_out), "![](performance_by_distance.png)\n"]

    # 4) is the observed OoD accuracy what distance alone predicts? -----------------
    ea = expected_accuracy(te.nn_sim.values, (te.pred == te.y).values, od.nn_sim.values,
                           min(a.B, 2000), rng)
    obs_acc = float((od.pred == od.y).mean())
    extrap_min = float(np.mean(od.nn_sim < te.nn_sim.min()))
    extrap_p5 = float(np.mean(od.nn_sim < te.nn_sim.quantile(0.05)))
    if ea is None:
        parts.append("Expected-accuracy check skipped: the model makes no (or only) errors on the test set.\n")
    else:
        tab = pd.DataFrame([
            {"reference model": "distance-agnostic (OoD accuracy = test accuracy)",
             "expected OoD accuracy": f"{ea['const_expected']:.3f}",
             "95% prediction interval": f"[{ea['const_pi'][0]:.3f}, {ea['const_pi'][1]:.3f}]",
             "P(acc <= observed)": f"{np.mean(ea['const_sims'] <= obs_acc):.3f}"},
            {"reference model": "logistic P(correct given nn-similarity), fitted on test",
             "expected OoD accuracy": f"{ea['expected']:.3f}",
             "95% prediction interval": f"[{ea['pi'][0]:.3f}, {ea['pi'][1]:.3f}]",
             "P(acc <= observed)": f"{np.mean(ea['sims'] <= obs_acc):.3f}"}])
        parts += ["### Expected vs. observed OoD accuracy\n",
                  f"Observed OoD accuracy: {obs_acc:.3f} ({int((od.pred == od.y).sum())}/{len(od)}). "
                  f"Slope of the correctness model: {ea['slope']:.2f} per unit similarity "
                  f"(bootstrap 95% CI [{ea['slope_ci'][0]:.2f}, {ea['slope_ci'][1]:.2f}]). "
                  "Prediction intervals include parameter uncertainty (bootstrap over test items) and sampling "
                  f"noise for n={len(od)}. Accuracy is used because correctness is a per-item quantity.\n",
                  md_table(tab),
                  f"Extrapolation: {extrap_min:.0%} of OoD items are less similar to the training data than any test "
                  f"item, {extrap_p5:.0%} lie below the test set's 5th percentile. If these shares are large, the "
                  "distance model is extrapolating and should be reported as indicative only.\n"]
    ctx.section("Distribution shift and leakage", *parts)


# ----------------------------------------------------------------------- baselines
def run_baselines(ctx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    a = ctx.args
    tr, te, od = ctx.sets["train"], ctx.sets["test"], ctx.sets["ood"]
    rng = np.random.default_rng(a.seed)
    systems = {"model": {"test": te.pred.values, "ood": od.pred.values}}
    lex = Lexicon(a.lexicon)
    systems["lexicon"] = {n: lex.predict(ctx.sets[n].text) for n in ("test", "ood")}
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=3000)).fit(tr.text, tr.y)
    systems["tfidf_lr"] = {n: tfidf.predict(ctx.sets[n].text) for n in ("test", "ood")}
    notes = ["- `lexicon`: predicts Non-Knowledge if any marker from the lexicon occurs "
             f"({'custom: ' + a.lexicon if a.lexicon else 'built-in generic English list'}).",
             "- `tfidf_lr`: word 1-2-gram TF-IDF + logistic regression trained on the training set."]

    if a.llm_model:
        client = LLMClient(a, ctx.cache_dir / "llm_answers.json")
        pred_te = np.full(len(te), -9)
        if a.llm_test_sample and a.llm_test_sample < len(te):
            frac = a.llm_test_sample / len(te)
            sel = np.concatenate([rng.choice(np.flatnonzero(te.y.values == c),
                                             max(1, round(frac * np.sum(te.y.values == c))), replace=False)
                                  for c in (0, 1)])
        else:
            sel = np.arange(len(te))
        log.info("LLM baseline: %d test + %d OoD items", len(sel), len(od))
        pred_te[sel] = client.classify(te.text.values[sel].tolist())
        pred_ood = client.classify(od.text.tolist())
        n_unparsed = {"test": int(np.sum(pred_te[sel] == -1)), "ood": int(np.sum(pred_ood == -1))}
        # unparseable answers count as errors (conservative)
        m = pred_te == -1; pred_te[m] = 1 - te.y.values[m]
        m = pred_ood == -1; pred_ood[m] = 1 - od.y.values[m]
        systems["llm_zero_shot"] = {"test": pred_te, "ood": pred_ood}
        notes.append(f"- `llm_zero_shot`: `{a.llm_model}` via {a.llm_base_url}, temperature 0, prompt hash "
                     f"`{client.prompt_hash}`; {len(sel)} stratified test items; unparseable answers "
                     f"(test {n_unparsed['test']}, OoD {n_unparsed['ood']}) counted as errors.")
        if sum(n_unparsed.values()) > 0.1 * (len(sel) + len(od)):
            log.warning("many unparseable LLM answers; raise --llm-max-tokens for reasoning models")
    ctx.baseline_preds = systems

    avail = {s: {n: P[n] != -9 for n in ("test", "ood")} for s, P in systems.items()}
    rows = []
    for s, P in systems.items():
        row = {"system": s}
        pts = {}
        for n in ("test", "ood"):
            df, msk = ctx.sets[n], avail[s][n]
            for m in ("f1_pos", "balanced_acc"):
                v = metric_ci(df.y.values[msk], P[n][msk], m, a.B, a.seed)
                row[f"{m} {n}"] = fmt(*v); pts[(m, n)] = v[0]
        row["drop f1_pos"] = f"{pts[('f1_pos', 'test')] - pts[('f1_pos', 'ood')]:+.3f}"
        row["drop bal_acc"] = f"{pts[('balanced_acc', 'test')] - pts[('balanced_acc', 'ood')]:+.3f}"
        rows.append(row)
    perf = pd.DataFrame(rows)

    comp = []
    for s in [s for s in systems if s != "model"]:
        row = {"baseline": s}
        for n in ("test", "ood"):
            msk, y = avail[s][n], ctx.sets[n].y.values
            n01, n10, p = mcnemar_exact(y[msk], systems["model"][n][msk], systems[s][n][msk])
            row[f"{n}: model-only correct / baseline-only correct"] = f"{n01} / {n10}"
            row[f"_p_{n}"] = p
        mt = avail[s]["test"]
        for m in ("f1_pos", "balanced_acc"):
            lo, hi = drop_difference_ci(te.y.values[mt], systems["model"]["test"][mt], systems[s]["test"][mt],
                                        od.y.values, systems["model"]["ood"], systems[s]["ood"], m, a.B, rng)
            dm = point(te.y.values[mt], systems["model"]["test"][mt])[m] - point(od.y.values, systems["model"]["ood"])[m]
            db = point(te.y.values[mt], systems[s]["test"][mt])[m] - point(od.y.values, systems[s]["ood"])[m]
            row[f"drop(model) - drop(baseline), {m}"] = fmt(dm - db, lo, hi)
        comp.append(row)
    comp = pd.DataFrame(comp)
    for n in ("test", "ood"):
        comp[f"McNemar p {n} (Holm)"] = [f"{p:.4f}" for p in holm(comp[f"_p_{n}"])]
    comp = comp[[c for c in comp.columns if not c.startswith("_p_")]]

    for n in ("test", "ood"):
        pd.DataFrame({"text": ctx.sets[n].text, "label": ctx.sets[n].y,
                      **{f"pred_{s}": P[n] for s, P in systems.items()}}).to_csv(
            ctx.out / f"baseline_predictions_{n}.csv", index=False)
    ctx.section("Baselines on the same items",
                "\n".join(notes) + "\n",
                "Performance (stratified BCa 95% CIs) and the point drop from test to OoD:\n",
                md_table(perf),
                "Paired comparisons against the model. McNemar uses only the discordant items, so it is far more "
                "powerful on 49 items than comparing scores across different datasets. The drop-difference CI answers "
                "'does our model lose more from test to OoD than this baseline?': a CI entirely below 0 means the model "
                "degrades less than the baseline, a CI entirely above 0 means it degrades more (model-specific shift sensitivity).\n",
                md_table(comp))


# ---------------------------------------------------------------------- invariance
NAME_POOL = ["dataHandler", "configLoader", "sessionCache", "eventQueue", "requestParser", "buildRunner"]
IDENT_RX = re.compile(r"`[^`\n]+`|\b[A-Za-z_][A-Za-z0-9_]*\(\)|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"
                      r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b|\b[a-z0-9]+(?:_[a-z0-9]+)+\b")
URL_RX = re.compile(r"https?://[^\s)>\]]*[^\s)>\].,;:!?'\"]")
PATH_RX = re.compile(r"(?<![\w:/])/(?:[\w.-]+/)+[\w-]+(\.\w{1,5})?")
FILE_RX = re.compile(r"\b[\w-]+\.(py|java|kt|js|ts|jsx|tsx|c|cc|cpp|h|hpp|go|rs|rb|php|cs|sh|ya?ml|json|xml"
                     r"|toml|ini|cfg|gradle|md|txt|lock)\b")
UNCERTAIN_PREFIXES = ["I'm not sure, but ", "I might be wrong, but ", "Not sure if this is right, but "]


def _restyle(camel, original):
    core = original.strip("`")
    call = core.endswith("()")
    core = core[:-2] if call else core
    if "_" in core:
        new = re.sub(r"(?<!^)([A-Z])", r"_\1", camel).lower()
    elif core[:1].isupper():
        new = camel[0].upper() + camel[1:]
    else:
        new = camel
    new = new + "()" if call else new
    return f"`{new}`" if original.startswith("`") else new


def swap_identifiers(text, rng, lex):
    found = list(dict.fromkeys(m.group(0) for m in IDENT_RX.finditer(text)))
    if not found:
        return None
    pool = list(rng.permutation(NAME_POOL))
    mapping = {}
    for i, f in enumerate(found):
        new = _restyle(pool[i % len(pool)], f)
        mapping[f] = new if new != f else _restyle(pool[(i + 1) % len(pool)], f)
    return IDENT_RX.sub(lambda m: mapping[m.group(0)], text)


def swap_numbers(text, rng, lex):
    if not re.search(r"\d", text):
        return None

    def rep(m):
        s = m.group(0)
        while True:
            n = "".join(str(rng.integers(0, 10)) for _ in s)
            if len(s) > 1 and n[0] == "0":
                n = str(rng.integers(1, 10)) + n[1:]
            if n != s:
                return n
    return re.sub(r"\d+", rep, text)


def swap_urls_paths(text, rng, lex):
    s = URL_RX.sub("https://example.org/docs/page", text)
    s = PATH_RX.sub(lambda m: "/opt/project/src/main" + (m.group(1) or ""), s)
    s = FILE_RX.sub(lambda m: f"module_a.{m.group(1)}", s)
    return s if s != text else None


def typo(text, rng, lex):
    protected = lex.spans(text) + [(m.start(), m.end()) for rx in (IDENT_RX, URL_RX, PATH_RX, FILE_RX)
                                   for m in rx.finditer(text)]
    cands = [(m.start(), m.end()) for m in re.finditer(r"\b[A-Za-z]{5,}\b", text)
             if not any(s < m.end() and m.start() < e for s, e in protected)]
    if not cands:
        return None
    s, e = cands[rng.integers(len(cands))]
    w = text[s:e]
    i = int(rng.integers(1, len(w) - 2))
    w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return None if w2 == w else text[:s] + w2 + text[e:]


def lowercase(text, rng, lex):
    return text.lower() if text.lower() != text else None


def add_uncertainty(text, rng, lex):
    t = text.lstrip()
    if len(t) > 1 and t[0].isupper() and t[1].islower() and not t.startswith("I "):
        t = t[0].lower() + t[1:]
    return UNCERTAIN_PREFIXES[rng.integers(len(UNCERTAIN_PREFIXES))] + t


def remove_markers(text, rng, lex):
    return lex.remove(text)


INV_TESTS = {"identifier_swap": swap_identifiers, "number_swap": swap_numbers,
             "url_path_swap": swap_urls_paths, "typo": typo, "lowercase": lowercase}
# name -> (function, gold label the test applies to, expected direction of p(Non-Knowledge))
DIR_TESTS = {"add_uncertainty_prefix": (add_uncertainty, 0, +1),
             "remove_uncertainty_markers": (remove_markers, 1, -1)}


def run_invariance(ctx):
    a = ctx.args
    lex = Lexicon(a.lexicon)
    rows, examples = [], []
    for name in ("test", "ood"):
        df = ctx.sets[name]
        items = np.arange(len(df))
        if name == "test" and a.inv_test_sample and a.inv_test_sample < len(df):
            items = np.sort(np.random.default_rng(a.seed).choice(len(df), a.inv_test_sample, replace=False))
        tests = [(t, fn, None, 0) for t, fn in INV_TESTS.items()] + \
                [(t, fn, lab, d) for t, (fn, lab, d) in DIR_TESTS.items()]
        for k, (tname, fn, lab, direction) in enumerate(tests):
            idx, pert = [], []
            for i in items:
                if lab is not None and df.y.values[i] != lab:
                    continue
                p = fn(df.text.values[i], np.random.default_rng([a.seed, int(i), k]), lex)
                if p is not None and p != df.text.values[i]:
                    idx.append(i); pert.append(p)
            idx = np.array(idx, int)
            if len(idx) == 0:
                rows.append({"set": name, "test": tname, "type": "DIR" if lab is not None else "INV", "n": 0})
                continue
            z = ctx.logits(pert)
            p_new = expit(z[:, 1] - z[:, 0]); pred_new = (p_new > 0.5).astype(int)
            p_old, pred_old, y = df.p_nk.values[idx], df.pred.values[idx], df.y.values[idx]
            dp = p_new - p_old
            row = {"set": name, "test": tname, "type": "DIR" if lab is not None else "INV", "n": len(idx)}
            if lab is None:
                k_flip = int(np.sum(pred_new != pred_old))
                lo, hi = cp_interval(k_flip, len(idx))
                row.update({"prediction flip rate": fmt(k_flip / len(idx), lo, hi),
                            "mean abs. Δp(NK)": f"{np.mean(np.abs(dp)):.3f}",
                            "accuracy before → after": f"{np.mean(pred_old == y):.3f} → {np.mean(pred_new == y):.3f}"})
                flagged = pred_new != pred_old
            else:
                k_dir = int(np.sum(dp * direction > 0))
                lo, hi = cp_interval(k_dir, len(idx))
                target = 1 if direction > 0 else 0
                cand = pred_old != target
                row.update({"moves in expected direction": fmt(k_dir / len(idx), lo, hi),
                            "median Δp(NK)": f"{np.median(dp):+.3f}",
                            "flips to expected label": f"{int(np.sum(pred_new[cand] == target))}/{int(cand.sum())}"})
                flagged = dp * direction < 0
            rows.append(row)
            for j in np.flatnonzero(flagged):
                examples.append({"set": name, "test": tname, "label": y[j], "original": df.text.values[idx[j]],
                                 "perturbed": pert[j], "p_nk_before": round(float(p_old[j]), 3),
                                 "p_nk_after": round(float(p_new[j]), 3)})
    res = pd.DataFrame(rows).fillna("")
    res.to_csv(ctx.out / "invariance_results.csv", index=False)
    pd.DataFrame(examples).to_csv(ctx.out / "invariance_examples.csv", index=False)
    inv, dr = res[res.type == "INV"], res[res.type == "DIR"]
    ctx.section("Invariance and directional tests",
                "INV: label-preserving perturbations; a robust model should rarely change its prediction. "
                "Compare flip rates between test and OoD, since higher OoD flip rates indicate brittleness on the new domain. "
                "DIR: adding an explicit uncertainty prefix to Knowledge items should raise p(Non-Knowledge); removing "
                "lexicon markers from Non-Knowledge items should lower it. These test sensitivity to the intended cue, "
                "not correctness (a prefixed statement need not become Non-Knowledge under your guideline), and "
                "marker removal can leave ungrammatical text. Flipped INV cases and DIR cases moving the wrong way "
                "are in `invariance_examples.csv`.\n",
                md_table(inv[[c for c in inv.columns if c and inv[c].astype(str).str.len().sum() > 0]]),
                md_table(dr[[c for c in dr.columns if c and dr[c].astype(str).str.len().sum() > 0]]))


# -------------------------------------------------------------------------- errors
def run_errors(ctx):
    a = ctx.args
    od, tr = ctx.sets["ood"], ctx.sets["train"]
    es, ei = ctx.nn("ood")
    lex = Lexicon(a.lexicon)
    df = pd.DataFrame({
        "item_id": np.arange(len(od)), "text": od.text, "label": od.y, "pred": od.pred,
        "error_type": np.where(od.y == od.pred, "",
                               np.where(od.y == 1, "FN (missed Non-Knowledge)", "FP (false Non-Knowledge)")),
        "p_non_knowledge": od.p_nk.round(3), "nn_sim_train": np.round(es, 3),
        "nearest_train_text": tr.text.values[ei], "nearest_train_label": tr.y.values[ei],
        "lexicon_hits": ["; ".join(lex.hits(t)) for t in od.text]})
    for s, P in ctx.baseline_preds.items():
        if s != "model":
            df[f"pred_{s}"] = P["ood"]
    df.to_csv(ctx.out / "ood_all_items.csv", index=False)
    err = df[df.error_type != ""].copy()
    err["error_category"], err["notes"] = "", ""
    err.to_csv(ctx.out / "ood_errors_for_coding.csv", index=False)

    parts = [f"{len(err)} OoD errors ({(err.error_type.str.startswith('FN')).sum()} FN, "
             f"{(err.error_type.str.startswith('FP')).sum()} FP) exported to `ood_errors_for_coding.csv` with "
             "the model's probability, the nearest training item, lexicon hits and baseline predictions. "
             "Fill `error_category` (one code per row) and re-run with `--coded-errors <file>`. "
             f"{int(np.sum((od.p_nk > 0.3) & (od.p_nk < 0.7) & (od.y == od.pred)))} further items are correct but with "
             "p(Non-Knowledge) between 0.3 and 0.7 (see `ood_all_items.csv`).\n",
             "A possible starting codebook, to be adapted to your guideline: *gold label debatable*; *implicit "
             "non-knowledge (no explicit marker)*; *marker used rhetorically / not expressing non-knowledge*; "
             "*domain-specific vocabulary or style*; *other model failure*.\n"]
    shared = [{"baseline": s, "also wrong on the model's errors": f"{int(np.sum(err[f'pred_{s}'] != err.label))}/{len(err)}"}
              for s in ctx.baseline_preds if s != "model"]
    if shared:
        parts += ["Errors shared with baselines (errors most baselines share point to hard or ambiguous items rather "
                  "than model-specific failures):\n", md_table(pd.DataFrame(shared))]
    if a.coded_errors:
        coded = pd.read_csv(a.coded_errors)
        coded["error_category"] = coded["error_category"].fillna("(uncoded)")
        ct = pd.crosstab(coded.error_category, coded.error_type, margins=True, margins_name="total").reset_index()
        parts += ["Coded errors:\n", md_table(ct)]
    ctx.section("OoD error analysis", *parts)


# ----------------------------------------------------------------------------- ATC
def atc_threshold(score, correct):
    """Threshold t with P_source(score < t) = source error rate (Garg et al., 2022).
    In the binary case max-softmax and negative entropy are both monotone in |margin|,
    so ATC-MC and ATC-NE coincide and temperature scaling does not change the estimate."""
    return np.quantile(score, 1 - correct.mean())


def run_atc(ctx):
    a = ctx.args
    rng = np.random.default_rng(a.seed)
    te, od = ctx.sets["test"], ctx.sets["ood"]
    s_te, c_te = np.abs(te.margin.values), (te.pred == te.y).values
    t = atc_threshold(s_te, c_te)
    if (~c_te).sum() < 10:
        log.warning("only %d test errors: the ATC threshold is poorly determined", int((~c_te).sum()))

    def boot(s_tgt, B):
        est_atc, est_ac = np.empty(B), np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(s_te), len(s_te)); j = rng.integers(0, len(s_tgt), len(s_tgt))
            est_atc[b] = np.mean(s_tgt[j] >= atc_threshold(s_te[i], c_te[i]))
            est_ac[b] = np.mean(expit(s_tgt[j]))
        return np.quantile(est_atc, [0.025, 0.975]), np.quantile(est_ac, [0.025, 0.975])

    B = min(a.B, 5000)
    s_od = np.abs(od.margin.values)
    obs = float((od.pred == od.y).mean())
    lo, hi = cp_interval(int((od.pred == od.y).sum()), len(od))
    ci_atc, ci_ac = boot(s_od, B)
    rows = [{"target": "OoD (labelled)", "n": len(od), "observed accuracy": fmt(obs, lo, hi),
             "ATC estimate": fmt(float(np.mean(s_od >= t)), *ci_atc),
             "average-confidence estimate": fmt(float(np.mean(expit(s_od))), *ci_ac),
             "predicted Non-Knowledge rate": f"{od.pred.mean():.3f}"}]
    parts = [f"Threshold fitted on the labelled test set (accuracy {c_te.mean():.3f}, {int((~c_te).sum())} errors; "
             "with very few test errors the threshold is poorly determined). Rows validate the "
             "estimator on the labelled OoD items first. Only if the ATC estimate lands near the observed accuracy "
             "is its estimate on unlabelled data credible. ATC estimates accuracy, not F1, and assumes the "
             "shift does not change which items the model is confidently wrong on.\n"]
    if "unlabeled" in ctx.sets:
        ul = ctx.sets["unlabeled"]
        s_ul = np.abs(ul.margin.values)
        ci_atc_u, ci_ac_u = boot(s_ul, B)
        rows.append({"target": "unlabelled pool", "n": len(ul), "observed accuracy": "n/a",
                     "ATC estimate": fmt(float(np.mean(s_ul >= t)), *ci_atc_u),
                     "average-confidence estimate": fmt(float(np.mean(expit(s_ul))), *ci_ac_u),
                     "predicted Non-Knowledge rate": f"{ul.pred.mean():.3f}"})
        ks_conf = stats.ks_2samp(s_od, s_ul)
        es_ul, _ = ctx.nn("unlabeled"); es_od, _ = ctx.nn("ood")
        ks_sim = stats.ks_2samp(es_od, es_ul)
        parts.append(f"Representativeness of the 49 labelled items for the unlabelled pool (two-sample KS): "
                     f"confidence |margin| D={ks_conf.statistic:.3f}, p={ks_conf.pvalue:.3f}; similarity to training "
                     f"data D={ks_sim.statistic:.3f}, p={ks_sim.pvalue:.3f}. Large D values mean the labelled sample "
                     "is not typical of the pool. Note that a non-significant KS test is weak evidence of similarity at n=49.\n")
    else:
        parts.append("No `--unlabeled` file given; re-run with one to estimate accuracy on the unlabelled pool.\n")
    ctx.section("Label-free performance estimation (ATC)", md_table(pd.DataFrame(rows)), *parts)


# ---------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("data")
    io.add_argument("--model", required=True)
    io.add_argument("--train", required=True, help="training CSV (texts + labels)")
    io.add_argument("--test", required=True, help="in-distribution test CSV")
    io.add_argument("--ood", required=True, help="labelled OoD CSV")
    io.add_argument("--unlabeled", help="unlabelled OoD CSV (text column only) for ATC")
    io.add_argument("--text-col", default=None, help="default: rephrased_statement > statement > text")
    io.add_argument("--label-col", default="final_label")
    io.add_argument("--label-map", action="append", metavar="SET:FROM=TO,...",
                    help="e.g. --label-map ood:1=0,2=1 (repeatable; sets: train, test, ood)")
    io.add_argument("--outdir", default="robustness_results")
    io.add_argument("--steps", default=",".join(ALL_STEPS))
    md = p.add_argument_group("classifier")
    md.add_argument("--device", default="auto")
    md.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="default matches the evaluation script")
    md.add_argument("--logit-indices", default="0,1", type=lambda s: tuple(int(x) for x in s.split(",")),
                    help="(Knowledge, Non-Knowledge) logit columns; '0,2' for a 3-class model")
    md.add_argument("--batch-size", type=int, default=64)
    md.add_argument("--max-length", type=int, default=512)
    sh = p.add_argument_group("shift")
    sh.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2",
                    help="sentence-transformers model, 'classifier' (mean-pooled fine-tuned model) or 'tfidf'")
    sh.add_argument("--dup-emb", type=float, default=0.95)
    sh.add_argument("--dup-char", type=float, default=0.90)
    sh.add_argument("--n-bins", type=int, default=5)
    sh.add_argument("--auc-reps", type=int, default=20)
    bl = p.add_argument_group("baselines")
    bl.add_argument("--lexicon", help="file with one regex per line (replaces the built-in list)")
    bl.add_argument("--llm-model", help="enables the zero-shot LLM baseline")
    bl.add_argument("--llm-base-url", default="http://localhost:8080/v1",
                    help="OpenAI-compatible endpoint (llama.cpp default; Ollama: http://localhost:11434/v1)")
    bl.add_argument("--llm-guideline", help="annotation guideline text inserted into the prompt")
    bl.add_argument("--llm-prompt", help="optional prompt template containing {guideline}")
    bl.add_argument("--llm-max-tokens", type=int, default=32, help="raise for reasoning models")
    bl.add_argument("--llm-timeout", type=float, default=120)
    bl.add_argument("--llm-test-sample", type=int, default=500, help="0 = full test set")
    ot = p.add_argument_group("other")
    ot.add_argument("--inv-test-sample", type=int, default=1000, help="0 = full test set")
    ot.add_argument("--coded-errors", help="coded ood_errors_for_coding.csv to summarise")
    ot.add_argument("--B", type=int, default=10000, help="bootstrap replicates")
    ot.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.llm_model and not args.llm_guideline:
        p.error("--llm-model requires --llm-guideline")
    return args


def main(argv=None, classifier=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}")
    ctx = Ctx(args, classifier)
    maps = parse_label_maps(args.label_map)
    ctx.sets["train"] = load_set(args.train, "train", args, maps, strict=False)
    ctx.sets["test"] = load_set(args.test, "test", args, maps)
    ctx.sets["ood"] = load_set(args.ood, "ood", args, maps)
    if args.unlabeled:
        ctx.sets["unlabeled"] = load_set(args.unlabeled, "unlabeled", args, maps, labelled=False)
    ctx.predict_all()
    run_overview(ctx)
    runners = {"shift": run_shift, "baselines": run_baselines, "invariance": run_invariance,
               "errors": run_errors, "atc": run_atc}
    for s in ALL_STEPS:  # fixed order: errors uses baseline predictions
        if s in steps:
            log.info("=== step: %s", s)
            runners[s](ctx)
    cfg = {k: v for k, v in vars(args).items()}
    header = ["# Robustness analysis report\n",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Positive class = Non-Knowledge (1).\n",
              "<details><summary>Configuration</summary>\n\n```json\n" + json.dumps(cfg, indent=2, default=str)
              + "\n```\n</details>\n"]
    (ctx.out / "report.md").write_text("\n".join(header + ctx.report), encoding="utf-8")
    log.info("report written to %s", ctx.out / "report.md")
    return ctx


if __name__ == "__main__":
    main()
