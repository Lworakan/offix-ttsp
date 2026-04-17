"""
ms_225 — TTSP (Test-Time Subject Probe): confidence-weighted mixture of
L/R-anchor and SIT-anchor experts, with gate computed FROM the held-out
subject's own unlabelled frames.

Design (novel 2026 idea):
  - Train two E1i models per fold: one with L/R anchor, one with SIT anchor.
  - At test time, for each test window, compute each expert's entropy.
  - Expert with lower mean entropy on the WHOLE held-out subject's frames
    is selected (subject-level gate).
  - Final softmax = g * p_LR + (1 - g) * p_SIT where g ∈ [0, 1] reflects
    which expert is more confident *on this subject*.

Variants:
  ttsp_subj   — single scalar gate per subject (from mean entropy)
  ttsp_sample — per-sample gate (using sample-level entropy)
  ttsp_oracle — per-subject oracle gate (upper bound for comparison)
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import warnings; warnings.filterwarnings("ignore")

from ms_140_common import CLASSES_4
from ms_62_day2_targeted import compute_subject_bone_reference
from ms_114_ecgf_xs import set_seed
from ms_173_final_9subj import (
    SUBJECTS_9, load_nom_segments, compute_lr_anchors, Dataset9Subj,
    FocalLoss, evaluate_model, TRAIN_STRIDE, EVAL_STRIDE, EPOCHS, BATCH,
    DEVICE, OUT_DIR,
)
from ms_176_offset_arch import SkelWithOffset
from ms_204_sit_anchor import compute_sit_anchors


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def entropy_np(p, axis=-1, eps=1e-9):
    return -np.sum(p * np.log(p + eps), axis=axis)


def train_one(anchors, train_seg, val_seg, test_seg, brefs, seed):
    train_ds = Dataset9Subj(train_seg, brefs, anchors, stride=TRAIN_STRIDE,
                             augment=True, use_anchor=True, feature_mode="full")
    val_ds = Dataset9Subj(val_seg, brefs, anchors, stride=EVAL_STRIDE,
                           augment=False, use_anchor=True, feature_mode="full")
    test_ds = Dataset9Subj(test_seg, brefs, anchors, stride=EVAL_STRIDE,
                            augment=False, use_anchor=True, feature_mode="full")
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True, generator=g)
    model = SkelWithOffset(3).to(DEVICE)
    pg = [{"params": m.parameters(), "lr": 5e-4 if n == "skeleton" else 1e-3}
          for n, m in model.named_children() if list(m.parameters())]
    opt = torch.optim.AdamW(pg, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = FocalLoss(gamma=2.0)
    best_val, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in train_ld:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            loss = crit(model(bd), batch["label"].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        vr = evaluate_model(model, val_ds)
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()

    def collect(ds):
        ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        lg, lb = [], []
        with torch.no_grad():
            for batch in ld:
                bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
                lg.append(model(bd).cpu().numpy())
                lb.append(batch["label"].numpy())
        return softmax_np(np.concatenate(lg)), np.concatenate(lb)

    return collect(test_ds)


def per_class(preds, labels):
    cm = confusion_matrix(labels, preds, labels=list(range(4)))
    pc = {c: float(cm[i, i] / max(cm[i].sum(), 1) * 100) for i, c in enumerate(CLASSES_4)}
    return {"macro": float(np.mean(list(pc.values()))), "per_class": pc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    print(f"ms_225 TTSP  seed={args.seed}", flush=True)

    all_segs, rows = load_nom_segments()
    brefs = {s: compute_subject_bone_reference(rows[s]) for s in SUBJECTS_9 if s in rows}
    lr_anchors = compute_lr_anchors(all_segs)
    sit_anchors = compute_sit_anchors(all_segs, n=25)

    fold_results = {"lr_only": [], "sit_only": [], "ttsp_subj": [],
                    "ttsp_sample": [], "ttsp_oracle": []}
    t0 = time.time()

    for fi, held_out in enumerate(SUBJECTS_9):
        if held_out not in brefs: continue
        train_pool = [s for s in SUBJECTS_9 if s != held_out and s in brefs]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]
        train_seg = {k: v for k, v in all_segs.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in all_segs.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in all_segs.items() if k[0] == held_out}
        print(f"\n=== fold {fi+1}/9: {held_out} ===", flush=True)

        p_LR, labels = train_one(lr_anchors, train_seg, val_seg, test_seg,
                                    brefs, args.seed + fi)
        p_SIT, _ = train_one(sit_anchors, train_seg, val_seg, test_seg,
                                brefs, args.seed + fi + 1000)

        # Entropies
        H_LR = entropy_np(p_LR)    # (N,)
        H_SIT = entropy_np(p_SIT)  # (N,)

        # Subject-level gate: compare mean entropy (LOWER entropy = more confident)
        g_subj = 1.0 if H_LR.mean() < H_SIT.mean() else 0.0
        p_subj = g_subj * p_LR + (1 - g_subj) * p_SIT

        # Per-sample gate: soft gate = softmax(-entropy)
        e_LR, e_SIT = -H_LR, -H_SIT   # higher is more confident
        g_sample = np.exp(e_LR) / (np.exp(e_LR) + np.exp(e_SIT))
        p_sample = g_sample[:, None] * p_LR + (1 - g_sample[:, None]) * p_SIT

        # Oracle: per-subject, pick whichever has higher macro on test (upper bound)
        r_LR = per_class(p_LR.argmax(1), labels)
        r_SIT = per_class(p_SIT.argmax(1), labels)
        g_oracle = 1.0 if r_LR["macro"] > r_SIT["macro"] else 0.0
        p_oracle = g_oracle * p_LR + (1 - g_oracle) * p_SIT

        r_subj = per_class(p_subj.argmax(1), labels)
        r_samp = per_class(p_sample.argmax(1), labels)
        r_orac = per_class(p_oracle.argmax(1), labels)

        np.savez(OUT_DIR / f"ms225_predictions_{held_out}.npz",
                 labels=labels, p_LR=p_LR, p_SIT=p_SIT,
                 p_subj=p_subj, p_sample=p_sample, p_oracle=p_oracle,
                 g_subj=np.float32(g_subj), g_sample=g_sample.astype(np.float32))

        for name, res in [("lr_only", r_LR), ("sit_only", r_SIT),
                           ("ttsp_subj", r_subj), ("ttsp_sample", r_samp),
                           ("ttsp_oracle", r_orac)]:
            entry = {"held_out": held_out, "test": res}
            if name == "ttsp_subj": entry["gate"] = g_subj
            if name == "ttsp_sample": entry["gate_mean"] = float(g_sample.mean())
            fold_results[name].append(entry)

        print(f"  LR   : fh={r_LR['per_class']['forward_head']:3.0f} sit={r_LR['per_class']['sit_straight']:3.0f} "
              f"L={r_LR['per_class']['left_leaning']:3.0f} R={r_LR['per_class']['right_leaning']:3.0f}  macro={r_LR['macro']:.1f} H̄={H_LR.mean():.3f}", flush=True)
        print(f"  SIT  : fh={r_SIT['per_class']['forward_head']:3.0f} sit={r_SIT['per_class']['sit_straight']:3.0f} "
              f"L={r_SIT['per_class']['left_leaning']:3.0f} R={r_SIT['per_class']['right_leaning']:3.0f}  macro={r_SIT['macro']:.1f} H̄={H_SIT.mean():.3f}", flush=True)
        print(f"  ttsp_subj(g={g_subj:.0f}): fh={r_subj['per_class']['forward_head']:3.0f} sit={r_subj['per_class']['sit_straight']:3.0f} "
              f"L={r_subj['per_class']['left_leaning']:3.0f} R={r_subj['per_class']['right_leaning']:3.0f}  macro={r_subj['macro']:.1f}", flush=True)
        print(f"  ttsp_samp: fh={r_samp['per_class']['forward_head']:3.0f} sit={r_samp['per_class']['sit_straight']:3.0f} "
              f"L={r_samp['per_class']['left_leaning']:3.0f} R={r_samp['per_class']['right_leaning']:3.0f}  macro={r_samp['macro']:.1f}  ḡ={g_sample.mean():.2f}", flush=True)

    print(f"\n{'='*70}\nTTSP SUMMARY\n{'='*70}", flush=True)
    for key in fold_results:
        folds = fold_results[key]
        m = np.mean([f["test"]["macro"] for f in folds])
        no_zero = sum(1 for f in folds if all(f["test"]["per_class"][c] >= 70 for c in CLASSES_4))
        all90 = sum(1 for f in folds if all(f["test"]["per_class"][c] >= 90 for c in CLASSES_4))
        pan = [f for f in folds if f["held_out"] == "pan"][0]["test"]["per_class"]
        muk = [f for f in folds if f["held_out"] == "mukrop"][0]["test"]["per_class"]
        print(f"  {key:>15}: macro={m:.1f}%  no_zero={no_zero}/9  all≥90={all90}/9  "
              f"pan_fh={pan['forward_head']:.0f}  muk_R={muk['right_leaning']:.0f}", flush=True)

    with open(OUT_DIR / "ms225_ttsp.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved.  Runtime: {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
