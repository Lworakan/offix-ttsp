"""
Bootstrap 95% CIs and McNemar exact tests for paper Table I.

Loads per-sample predictions for the four reported methods --- Skel-only
(ms_173 E1a), lr_only (ms_225 p_LR), ttsp_subj (ms_225 p_subj), and the
calibration-validated TTSP-selector (ms_227) --- on the same eval split
(test minus the first 5 windows per class used as the calibration
validator), then reports cohort macro-F1 with 10 000-sample bootstrap CIs
and McNemar's exact test for each method against Skel-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ms_140_common import CLASSES_4

PRED_DIR = ROOT / "outputs"
SUBJECTS = ["fu", "mukrop", "nonny", "boom", "peemai", "pan", "namoon", "mai", "money"]
CAL_K = 5
B = 10_000
SEED = 42
PRIORITY = ("skel_only", "lr_only", "ms224_e1i", "ttsp_subj")


def macro(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def per_class(y, p):
    return {CLASSES_4[c]: float((p[y == c] == c).mean()) if (y == c).any() else 0.0
            for c in range(4)}


def cal_eval_split(labels, k=CAL_K):
    cal = np.array(sorted(int(i) for c in range(4) for i in np.where(labels == c)[0][:k]),
                   dtype=np.int64)
    mask = np.zeros(len(labels), dtype=bool); mask[cal] = True
    return cal, np.where(~mask)[0]


def load_per_subject(subj):
    """Returns (labels, {method -> argmax preds}) all aligned to ms_225's window grid."""
    z225 = np.load(PRED_DIR / f"ms225_predictions_{subj}.npz")
    labels = z225["labels"]
    preds = {
        "lr_only":   z225["p_LR"].argmax(1),
        "ttsp_subj": z225["p_subj"].argmax(1),
    }
    f173 = PRED_DIR / f"ms173_predictions_E1a_{subj}.npz"
    if f173.exists():
        z = np.load(f173)
        n = min(len(labels), len(z["labels"]))
        if (z["labels"][:n] == labels[:n]).all():
            preds["skel_only"] = z["preds"][:n]
    f224 = PRED_DIR / f"ms224_predictions_{subj}.npz"
    if f224.exists():
        z = np.load(f224)
        if (z["full_labels"] == labels).all():
            preds["ms224_e1i"] = z["full_p_e1i"].argmax(1)
    return labels, preds


def selector_pred(preds, cal_idx, labels):
    cal_accs = {c: float((preds[c][cal_idx] == labels[cal_idx]).mean())
                for c in PRIORITY if c in preds}
    best = max(cal_accs.values())
    return next(c for c in PRIORITY if c in cal_accs and cal_accs[c] == best)


def oracle_pred(preds, eval_idx, labels):
    return max(preds, key=lambda c: macro(labels[eval_idx], preds[c][eval_idx]))


def bootstrap(folds, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(folds)
    macros, all90s = [], []
    for _ in range(B):
        sidx = rng.integers(0, n, size=n)
        m, count = [], 0
        for k in sidx:
            y, p = folds[k]
            f = rng.integers(0, len(y), size=len(y))
            m.append(macro(y[f], p[f]))
            if all(v >= 0.9 for v in per_class(y[f], p[f]).values()):
                count += 1
        macros.append(np.mean(m)); all90s.append(count)
    return (np.mean(macros), np.percentile(macros, 2.5), np.percentile(macros, 97.5),
            np.mean(all90s))


def mcnemar(folds_a, folds_b):
    b = c = 0
    for (yA, pA), (yB, pB) in zip(folds_a, folds_b):
        n = min(len(yA), len(yB))
        a_ok = (pA[:n] == yA[:n]); b_ok = (pB[:n] == yB[:n])
        b += int(np.sum(a_ok & ~b_ok))
        c += int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def main():
    method_folds = {m: [] for m in
                    ["skel_only", "lr_only", "ttsp_subj", "ms224_e1i",
                     "ttsp_selector", "oracle_ub"]}
    selector_choices = []

    for subj in SUBJECTS:
        labels, preds = load_per_subject(subj)
        cal, eval_ = cal_eval_split(labels)

        for m in ("skel_only", "lr_only", "ttsp_subj", "ms224_e1i"):
            if m not in preds: continue
            p = preds[m]
            n = min(len(p), len(labels))
            mask = eval_[eval_ < n]
            method_folds[m].append((labels[mask], p[mask]))

        chose = selector_pred(preds, cal, labels)
        method_folds["ttsp_selector"].append((labels[eval_], preds[chose][eval_]))
        selector_choices.append((subj, chose))

        oracle = oracle_pred(preds, eval_, labels)
        method_folds["oracle_ub"].append((labels[eval_], preds[oracle][eval_]))

    print(f"\nCohort eval-split macro-F1 with bootstrap 95% CI (B={B})\n")
    print(f"{'method':<15} {'mean':>7} {'95% CI':>18} {'>=90':>5}")
    print("-" * 50)
    summary = {}
    for m, folds in method_folds.items():
        if not folds: continue
        mean_m, lo, hi, _ = bootstrap(folds)
        observed_all90 = sum(1 for y, p in folds
                             if all(v >= 0.9 for v in per_class(y, p).values()))
        per_fold = [macro(y, p) for y, p in folds]
        print(f"{m:<15} {mean_m*100:6.2f}%  [{lo*100:5.2f}, {hi*100:6.2f}]  "
              f"{observed_all90}/{len(folds)}")
        summary[m] = {
            "mean": float(mean_m),
            "ci_low": float(lo), "ci_high": float(hi),
            "all90_observed": observed_all90,
            "per_fold_macro": [float(x) for x in per_fold],
        }

    print(f"\nMcNemar exact test vs skel_only:")
    summary["mcnemar_vs_skel"] = {}
    for m in ("lr_only", "ttsp_subj", "ttsp_selector", "ms224_e1i"):
        if not method_folds.get(m) or not method_folds.get("skel_only"): continue
        b, c, p = mcnemar(method_folds[m], method_folds["skel_only"])
        flag = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
        print(f"  {m:<15}: b={b:>3} c={c:>3} p={p:.2e} {flag}")
        summary["mcnemar_vs_skel"][m] = {"b": b, "c": c, "p": float(p)}

    summary["selector_choices"] = dict(selector_choices)
    (PRED_DIR / "ms302_table1_stats.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"\nsaved {PRED_DIR / 'ms302_table1_stats.json'}")


if __name__ == "__main__":
    main()
