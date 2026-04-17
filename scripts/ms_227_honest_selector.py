"""
Calibration-validated method selector for TTSP.

For each held-out subject we already have per-sample softmaxes from the
trained TTSP entropy gate (ms_225), the single LR-anchored expert (ms_225
p_LR), and the 1-shot prototype hybrid (ms_224). The selector treats the
first 5 windows per class as the user's calibration recording (which is
labelled by construction --- the user pressed L/R/sit during the wizard)
and routes them to whichever candidate scored highest on that recording.
Ties broken by Occam priority: lr_only > 1-shot > entropy gate.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ms_140_common import CLASSES_4

PRED_DIR = ROOT / "outputs"
SUBJECTS = ["fu", "mukrop", "nonny", "boom", "peemai", "pan", "namoon", "mai", "money"]
CAL_K = 5
PRIORITY = ("skel_only", "lr_only", "ms224_hyb", "ttsp_subj")


def macro_recall(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def per_class(y, p):
    return {CLASSES_4[c]: float((p[y == c] == c).mean()) if (y == c).any() else 0.0
            for c in range(4)}


def cal_eval_split(labels, k=CAL_K):
    cal = sorted(int(i) for c in range(4) for i in np.where(labels == c)[0][:k])
    cal = np.asarray(cal, dtype=np.int64)
    mask = np.zeros(len(labels), dtype=bool); mask[cal] = True
    return cal, np.where(~mask)[0]


def load_candidates(subj):
    """Return {candidate -> argmax predictions over full test set}."""
    z = np.load(PRED_DIR / f"ms225_predictions_{subj}.npz")
    labels = z["labels"]
    out = {
        "ttsp_subj": z["p_subj"].argmax(1),
        "lr_only":   z["p_LR"].argmax(1),
    }
    f173 = PRED_DIR / f"ms173_predictions_E1a_{subj}.npz"
    if f173.exists():
        z173 = np.load(f173)
        n = min(len(labels), len(z173["labels"]))
        if (z173["labels"][:n] == labels[:n]).all() and n == len(labels):
            out["skel_only"] = z173["preds"]
    f224 = PRED_DIR / f"ms224_predictions_{subj}.npz"
    if f224.exists():
        z224 = np.load(f224)
        if (z224["full_labels"] == labels).all():
            out["ms224_hyb"] = z224["full_p_e1i"].argmax(1)
    return labels, out


def select(cal_accs):
    best = max(cal_accs.values())
    return next(c for c in PRIORITY if c in cal_accs and cal_accs[c] == best)


def main():
    print(f"ms_227 calibration-validated selector  (CAL_K={CAL_K} per class)\n")
    rows = []
    for subj in SUBJECTS:
        labels, cands = load_candidates(subj)
        cal, eval_ = cal_eval_split(labels)
        cal_accs = {name: float((p[cal] == labels[cal]).mean()) for name, p in cands.items()}
        eval_macros = {name: macro_recall(labels[eval_], p[eval_]) for name, p in cands.items()}
        chose = select(cal_accs)

        rows.append({"subject": subj, "chose": chose,
                     "cal_accs": cal_accs, "eval_macros": eval_macros,
                     "eval_pc_chose": per_class(labels[eval_], cands[chose][eval_])})

        cal_str = "  ".join(f"{n}: cal={a*100:.0f}|eval={eval_macros[n]*100:.1f}"
                            for n, a in cal_accs.items())
        print(f"  {subj:>8}  {cal_str}  -> CHOSE {chose}")

    selector_macros = [macro_recall(np.asarray([]), np.asarray([])) for _ in rows]  # placeholder
    selector_macros = [r["eval_macros"][r["chose"]] for r in rows]
    oracle_macros = [max(r["eval_macros"].values()) for r in rows]
    selector_all90 = sum(1 for r in rows if all(v >= 0.9 for v in r["eval_pc_chose"].values()))

    print(f"\nselector cohort macro = {np.mean(selector_macros)*100:.2f}%   "
          f"all-class>=90 = {selector_all90}/{len(rows)}")
    print(f"oracle UB cohort macro = {np.mean(oracle_macros)*100:.2f}%")

    out = {
        "cal_k": CAL_K,
        "candidate_priority": list(PRIORITY),
        "per_subject": rows,
        "cohort": {
            "selector_macro": float(np.mean(selector_macros)),
            "selector_all90": int(selector_all90),
            "oracle_macro": float(np.mean(oracle_macros)),
        },
    }
    (PRED_DIR / "ms227_honest_selector.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
