"""
ms_224 — Corrected 1-shot per-class calibration, 9-fold LOSO.

Design (matches ergonomic-clinic deployment practice):
  1. Train E1i (SkelWithOffset + L/R anchor) normally on 8 subjects.
  2. Held-out subject provides 1 random window per class (4 "shots" total)
     and those shots are REMOVED from the test set.
  3. Shots define per-subject per-class clinical-feature prototypes
     in raw (un-anchored) space.
  4. Nearest-centroid classifier p_proto = softmax(-d / T) where d[c] is
     euclidean distance from test clinical features to prototype c.
  5. Final = alpha * p_e1i + (1 - alpha) * p_proto.
  6. alpha is selected per-fold on the VAL subject (not the held-out) by
     grid search in [0, 0.1, ..., 1.0].
"""
import sys, json, time, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import warnings; warnings.filterwarnings("ignore")

from ms_140_common import CLASSES_4, WINDOW_T
from ms_62_day2_targeted import compute_subject_bone_reference
from ms_114_ecgf_xs import set_seed
from ms_173_final_9subj import (
    SUBJECTS_9, load_nom_segments, compute_lr_anchors, Dataset9Subj,
    FocalLoss, evaluate_model, TRAIN_STRIDE, EVAL_STRIDE, EPOCHS, BATCH,
    DEVICE, OUT_DIR,
)
from ms_176_offset_arch import SkelWithOffset


def softmax_np(x, axis=-1, T=1.0):
    x = x / T
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def train_e1i(train_seg, val_seg, brefs, anchors, seed):
    train_ds = Dataset9Subj(train_seg, brefs, anchors, stride=TRAIN_STRIDE,
                             augment=True, use_anchor=True, feature_mode="full")
    val_ds = Dataset9Subj(val_seg, brefs, anchors, stride=EVAL_STRIDE,
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
    return model, val_ds


def collect(model, ds):
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    lg, lb, ex = [], [], []
    with torch.no_grad():
        for batch in ld:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            lg.append(model(bd).cpu().numpy())
            lb.append(batch["label"].numpy())
            ex.append(batch["extra"].cpu().numpy())
    return np.concatenate(lg), np.concatenate(lb), np.concatenate(ex)


def per_class(preds, labels):
    cm = confusion_matrix(labels, preds, labels=list(range(4)))
    pc = {c: float(cm[i, i] / max(cm[i].sum(), 1) * 100) for i, c in enumerate(CLASSES_4)}
    return {"macro": float(np.mean(list(pc.values()))), "per_class": pc}


def get_raw_features(ds):
    """Pull raw (un-anchored) clinical 3-vec [sh_ear, HFD, CVA] per window (middle frame)."""
    raw = []
    for si, start in ds.windows:
        feats = ds.segments[si]["frames"][start + WINDOW_T // 2]["feats"]
        # feats stored are anchored; add back anchor to get raw
        subj = ds.segments[si]["subject"]
        anc = ds.anchors[subj] if hasattr(ds, 'anchors') else None
        # feats shape (4,) — [sh, neck, hfd, cva]; extract [sh, hfd, cva] = indices 0,2,3
        raw.append(np.array([feats[0], feats[2], feats[3]], dtype=np.float32))
    return np.array(raw, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shots", type=int, default=1, help="samples per class for calibration")
    args = ap.parse_args()
    set_seed(args.seed)
    print(f"ms_224 1-shot calibration v2  seed={args.seed}  shots={args.shots}", flush=True)

    all_segs, rows = load_nom_segments()
    brefs = {s: compute_subject_bone_reference(rows[s]) for s in SUBJECTS_9 if s in rows}
    anchors = compute_lr_anchors(all_segs)

    # We need raw features per window; augment Dataset9Subj with anchors attr
    Dataset9Subj.anchors = anchors

    fold_results = {"e1i_only": [], "proto_only": [], "hybrid": []}
    t0 = time.time()

    for fi, held_out in enumerate(SUBJECTS_9):
        if held_out not in brefs: continue
        train_pool = [s for s in SUBJECTS_9 if s != held_out and s in brefs]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]
        train_seg = {k: v for k, v in all_segs.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in all_segs.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in all_segs.items() if k[0] == held_out}
        print(f"\n=== fold {fi+1}/9: {held_out} (val={val_subj}) ===", flush=True)

        model, val_ds = train_e1i(train_seg, val_seg, brefs, anchors, args.seed + fi)

        # Build eval ds (full test, stride=1)
        test_ds = Dataset9Subj(test_seg, brefs, anchors, stride=EVAL_STRIDE,
                                augment=False, use_anchor=True, feature_mode="full")
        test_logits, test_labels, _ = collect(model, test_ds)
        test_sm = softmax_np(test_logits)
        test_raw = get_raw_features(test_ds)

        # Pick `shots` random windows per class from test_ds as 1-shot calibration
        rng = np.random.RandomState(args.seed + fi)
        shot_idx_per_class = {}
        for c in range(4):
            class_idx = np.where(test_labels == c)[0]
            if len(class_idx) == 0:
                shot_idx_per_class[c] = np.array([], dtype=int); continue
            shot_idx_per_class[c] = rng.choice(class_idx, size=min(args.shots, len(class_idx)), replace=False)

        all_shot_idx = np.concatenate(list(shot_idx_per_class.values())) if shot_idx_per_class else np.array([], dtype=int)
        eval_mask = np.ones(len(test_labels), dtype=bool); eval_mask[all_shot_idx] = False

        # Compute prototypes in raw clinical space
        protos = np.zeros((4, 3), dtype=np.float32)
        for c in range(4):
            if len(shot_idx_per_class[c]) > 0:
                protos[c] = test_raw[shot_idx_per_class[c]].mean(0)
            else:
                protos[c] = test_raw.mean(0)  # fallback

        # Prototype softmax on test-minus-shots
        eval_raw = test_raw[eval_mask]
        eval_labels = test_labels[eval_mask]
        eval_sm_e1i = test_sm[eval_mask]
        # Normalise per-feature by std so distances are scale-invariant
        std = test_raw.std(0, ddof=1) + 1e-6
        dists = np.linalg.norm((eval_raw[:, None, :] - protos[None, :, :]) / std, axis=-1)  # (N, 4)
        T = 1.0
        proto_sm = softmax_np(-dists, T=T)

        # Pick alpha by grid search on VAL subject
        val_logits, val_labels, _ = collect(model, val_ds)
        val_sm_e1i = softmax_np(val_logits)
        val_raw = get_raw_features(val_ds)
        # For val we don't have shots from val subject — skip alpha tuning on val;
        # use a SIMULATED 1-shot on val: pick 1 random sample per class from val,
        # compute val prototypes, then evaluate remaining val samples.
        val_shot_idx = {}
        for c in range(4):
            ci = np.where(val_labels == c)[0]
            if len(ci) == 0:
                val_shot_idx[c] = np.array([], dtype=int); continue
            val_shot_idx[c] = rng.choice(ci, size=min(args.shots, len(ci)), replace=False)
        val_all_shot = np.concatenate(list(val_shot_idx.values())) if val_shot_idx else np.array([], dtype=int)
        val_eval_mask = np.ones(len(val_labels), dtype=bool); val_eval_mask[val_all_shot] = False
        val_protos = np.zeros((4, 3), dtype=np.float32)
        for c in range(4):
            if len(val_shot_idx[c]) > 0:
                val_protos[c] = val_raw[val_shot_idx[c]].mean(0)
            else:
                val_protos[c] = val_raw.mean(0)
        val_eval_raw = val_raw[val_eval_mask]
        val_eval_labels = val_labels[val_eval_mask]
        val_eval_sm_e1i = val_sm_e1i[val_eval_mask]
        val_std = val_raw.std(0, ddof=1) + 1e-6
        val_dists = np.linalg.norm((val_eval_raw[:, None, :] - val_protos[None, :, :]) / val_std, axis=-1)
        val_proto_sm = softmax_np(-val_dists, T=T)

        best_alpha, best_macro = 0.5, -1
        for a in np.linspace(0, 1, 11):
            mix = a * val_eval_sm_e1i + (1 - a) * val_proto_sm
            preds = mix.argmax(1)
            r = per_class(preds, val_eval_labels)
            if r["macro"] > best_macro:
                best_macro, best_alpha = r["macro"], float(a)

        # Apply best alpha to test-minus-shots
        mix = best_alpha * eval_sm_e1i + (1 - best_alpha) * proto_sm
        r_hyb = per_class(mix.argmax(1), eval_labels)
        r_e1i = per_class(eval_sm_e1i.argmax(1), eval_labels)
        r_proto = per_class(proto_sm.argmax(1), eval_labels)

        # Dump per-sample predictions for downstream honest-selector / bootstrap analysis.
        np.savez(OUT_DIR / f"ms224_predictions_{held_out}.npz",
                 labels=eval_labels,
                 p_e1i=eval_sm_e1i,
                 p_proto=proto_sm,
                 p_hybrid=mix,
                 alpha=np.float32(best_alpha),
                 # full-test version (BEFORE removing shots), for selector use:
                 full_labels=test_labels,
                 full_p_e1i=test_sm,
                 shot_idx=np.array(all_shot_idx, dtype=np.int64))
        for name, res in [("e1i_only", r_e1i), ("proto_only", r_proto), ("hybrid", r_hyb)]:
            fold_results[name].append({
                "held_out": held_out, "test": res, "alpha": best_alpha,
                "shot_idx": {int(c): shot_idx_per_class[c].tolist() for c in shot_idx_per_class},
            })
        print(f"  E1i    : fh={r_e1i['per_class']['forward_head']:3.0f} sit={r_e1i['per_class']['sit_straight']:3.0f} "
              f"L={r_e1i['per_class']['left_leaning']:3.0f} R={r_e1i['per_class']['right_leaning']:3.0f}", flush=True)
        print(f"  proto  : fh={r_proto['per_class']['forward_head']:3.0f} sit={r_proto['per_class']['sit_straight']:3.0f} "
              f"L={r_proto['per_class']['left_leaning']:3.0f} R={r_proto['per_class']['right_leaning']:3.0f}", flush=True)
        print(f"  hybrid (α={best_alpha:.2f}): fh={r_hyb['per_class']['forward_head']:3.0f} sit={r_hyb['per_class']['sit_straight']:3.0f} "
              f"L={r_hyb['per_class']['left_leaning']:3.0f} R={r_hyb['per_class']['right_leaning']:3.0f}", flush=True)

    print(f"\n{'='*70}\n1-SHOT SUMMARY\n{'='*70}", flush=True)
    for key in fold_results:
        folds = fold_results[key]
        m = np.mean([f["test"]["macro"] for f in folds])
        no_zero = sum(1 for f in folds if all(f["test"]["per_class"][c] >= 70 for c in CLASSES_4))
        all90 = sum(1 for f in folds if all(f["test"]["per_class"][c] >= 90 for c in CLASSES_4))
        pan = [f for f in folds if f["held_out"] == "pan"][0]["test"]["per_class"]
        muk = [f for f in folds if f["held_out"] == "mukrop"][0]["test"]["per_class"]
        print(f"  {key:>12}: macro={m:.1f}%  no_zero={no_zero}/9  all≥90={all90}/9  "
              f"pan_fh={pan['forward_head']:.0f}  muk_R={muk['right_leaning']:.0f}", flush=True)

    with open(OUT_DIR / "ms224_oneshot_v2.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved to {OUT_DIR / 'ms224_oneshot_v2.json'}  Runtime: {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
