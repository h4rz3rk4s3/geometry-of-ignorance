"""
Uncertainty quantification for binary classification metrics on small eval sets.

Provides, for one or two sets of (y_true, y_pred):
  1. Stratified BCa bootstrap CIs (class counts held fixed, as in the design)
  2. Bayesian credible intervals (Jeffreys-Beta posteriors on recall and specificity)
  3. Exact Clopper-Pearson CIs for recall and specificity
  4. Permutation test against a no-skill classifier (Ojala & Garriga, 2010)
  5. Bootstrap CI for the difference in-distribution test minus OoD

Input CSVs need a true-label and a predicted-label column (0/1, positive class = 1).

Usage:
  python f1_uncertainty.py --ood revision/results/predictions_OoD.csv \
    --val revision/results/predictions_Ind.csv \
    --pred-col 22-cue-matcher
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

POS = 1
METRICS = ["f1_pos", "f1_macro", "f1_weighted", "balanced_acc", "mcc"]


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def metrics_from_counts(tp, fp, fn, tn):
    """All metrics from confusion counts; works elementwise on arrays.
    Zero-division yields 0 (same convention as sklearn's zero_division=0)."""
    tp, fp, fn, tn = (np.asarray(x, float) for x in (tp, fp, fn, tn))
    n_pos, n_neg = tp + fn, tn + fp
    f1_pos = _div(2 * tp, 2 * tp + fp + fn)
    f1_neg = _div(2 * tn, 2 * tn + fn + fp)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "f1_pos": f1_pos,
        "f1_macro": (f1_pos + f1_neg) / 2,
        "f1_weighted": _div(n_pos * f1_pos + n_neg * f1_neg, n_pos + n_neg),
        "balanced_acc": (_div(tp, n_pos) + _div(tn, n_neg)) / 2,
        "mcc": _div(tp * tn - fp * fn, mcc_den),
    }


def counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == POS) & (y_pred == POS))
    fp = np.sum((y_true != POS) & (y_pred == POS))
    fn = np.sum((y_true == POS) & (y_pred != POS))
    tn = np.sum((y_true != POS) & (y_pred != POS))
    return tp, fp, fn, tn


def stratified_bootstrap(y_true, y_pred, B, rng):
    """Resample positives and negatives separately (class counts fixed).
    Returns dict metric -> array of B bootstrap replicates."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pred_pos = y_pred[y_true == POS] == POS   # hit indicators among positives
    pred_neg = y_pred[y_true != POS] == POS   # false-alarm indicators among negatives
    tp = rng.choice(pred_pos, (B, pred_pos.size)).sum(1)
    fp = rng.choice(pred_neg, (B, pred_neg.size)).sum(1)
    return metrics_from_counts(tp, fp, pred_pos.size - tp, pred_neg.size - fp)


def jackknife(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.arange(len(y_true))
    reps = [metrics_from_counts(*counts(y_true[idx != i], y_pred[idx != i]))
            for i in idx]
    return {m: np.array([r[m] for r in reps]) for m in METRICS}


def bca_interval(theta, boot, jack, alpha=0.05):
    """Bias-corrected and accelerated interval (Efron, 1987).
    Mid-rank bias correction handles the discrete bootstrap distribution."""
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / boot.size
    if prop <= 0 or prop >= 1:  # degenerate: fall back to percentile
        return tuple(np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    z0 = stats.norm.ppf(prop)
    d = jack.mean() - jack
    den = 6 * np.sum(d ** 2) ** 1.5
    a = np.sum(d ** 3) / den if den > 0 else 0.0
    z = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return tuple(np.quantile(boot, adj))


def bayesian_interval(y_true, y_pred, draws, rng, alpha=0.05, prior=0.5):
    """Jeffreys-Beta posteriors on recall (TPR) and specificity (TNR),
    class counts fixed; metrics are pushed through the posterior draws."""
    tp, fp, fn, tn = counts(y_true, y_pred)
    n_pos, n_neg = tp + fn, tn + fp
    tpr = rng.beta(tp + prior, fn + prior, draws)
    fpr = rng.beta(fp + prior, tn + prior, draws)
    m = metrics_from_counts(n_pos * tpr, n_neg * fpr, n_pos * (1 - tpr), n_neg * (1 - fpr))
    return {k: tuple(np.quantile(v, [alpha / 2, 1 - alpha / 2])) for k, v in m.items()}


def permutation_test(y_true, y_pred, B, rng):
    """H0: predictions carry no information about labels. One-sided p-values."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    obs = metrics_from_counts(*counts(y_true, y_pred))
    perm = rng.permuted(np.tile(y_true, (B, 1)), axis=1)
    pp = y_pred[None, :] == POS
    tp = np.sum((perm == POS) & pp, 1)
    fp = np.sum((perm != POS) & pp, 1)
    fn = np.sum((perm == POS) & ~pp, 1)
    tn = np.sum((perm != POS) & ~pp, 1)
    null = metrics_from_counts(tp, fp, fn, tn)
    return {m: (1 + np.sum(null[m] >= obs[m])) / (B + 1) for m in METRICS}


def exact_rates(y_true, y_pred, alpha=0.05):
    tp, fp, fn, tn = counts(y_true, y_pred)
    out = {}
    for name, k, n in [("recall (TPR)", tp, tp + fn), ("specificity (TNR)", tn, tn + fp)]:
        ci = stats.binomtest(int(k), int(n)).proportion_ci(1 - alpha, method="exact")
        out[name] = (k, n, k / n, ci.low, ci.high)
    return out


def summarize(y_true, y_pred, name="set", B=20000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp, fp, fn, tn = counts(y_true, y_pred)
    print(f"\n=== {name}: n={len(y_true)}, pos={tp+fn}, neg={tn+fp} ===")
    print(f"Confusion: TP={tp} FN={fn} FP={fp} TN={tn}")
    for k, (kk, n, p, lo, hi) in exact_rates(y_true, y_pred, alpha).items():
        print(f"  {k:18s} {kk}/{n} = {p:.3f}  Clopper-Pearson {1-alpha:.0%} CI [{lo:.3f}, {hi:.3f}]")
    obs = metrics_from_counts(tp, fp, fn, tn)
    boot = stratified_bootstrap(y_true, y_pred, B, rng)
    jack = jackknife(y_true, y_pred)
    bayes = bayesian_interval(y_true, y_pred, B, rng, alpha)
    pvals = permutation_test(y_true, y_pred, B, rng)
    print(f"  {'metric':13s} {'point':>6s}   {'BCa bootstrap':>17s}   {'Bayesian (Jeffreys)':>19s}   {'perm. p':>7s}")
    for m in METRICS:
        lo, hi = bca_interval(float(obs[m]), boot[m], jack[m], alpha)
        blo, bhi = bayes[m]
        print(f"  {m:13s} {float(obs[m]):6.3f}   [{lo:.3f}, {hi:.3f}]    [{blo:.3f}, {bhi:.3f}]      {pvals[m]:.5f}")
    return boot


def compare(test, ood, B=20000, seed=1, alpha=0.05):
    """Percentile CI for (test - OoD); both sets resampled independently."""
    rng = np.random.default_rng(seed)
    bt = stratified_bootstrap(*test, B, rng)
    bo = stratified_bootstrap(*ood, B, rng)
    ot = metrics_from_counts(*counts(*test))
    oo = metrics_from_counts(*counts(*ood))
    print(f"\n=== Difference: test - OoD (percentile bootstrap, {1-alpha:.0%}) ===")
    for m in METRICS:
        d = bt[m] - bo[m]
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        print(f"  {m:13s} diff={float(ot[m]-oo[m]):+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")


def load(path, true_col, pred_col):
    df = pd.read_csv(path)
    y_true, y_pred = df[true_col].astype(int).to_numpy(), df[pred_col].astype(int).to_numpy()
    assert set(np.unique(y_true)) <= {0, 1} and set(np.unique(y_pred)) <= {0, 1}, \
        f"{path}: labels must be 0/1 after mapping"
    return y_true, y_pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test")
    ap.add_argument("--true-col", default="final_label")
    ap.add_argument("--pred-col", default="predicted_label")
    ap.add_argument("--B", type=int, default=20000)
    args = ap.parse_args()
    ood = load(args.ood, args.true_col, args.pred_col)
    val = load(args.val, args.true_col, args.pred_col)
    summarize(*ood, name="OoD", B=args.B)
    summarize(*val, name="Validation", B=args.B)
    if args.test:
        test = load(args.test, args.true_col, args.pred_col)
        summarize(*test, name="Test (in-distribution)", B=args.B)
        compare(test, ood, B=args.B)
