"""
ms_204 — E2a: Use SIT samples as calibration anchor instead of L+R average.

Root-cause finding (from ms_203 diagnostic): the L/R calibration anchor assumes
that lateral leanings are approximately neutral posture. For subjects whose L/R
recordings involve shoulder rotation (e.g., pan old session), the anchor lands
closer to the forward-head region, corrupting anchored features for all classes.

Sit_straight is explicitly the neutral protocol class. Using its mean as the
calibration anchor:
  - Guarantees the anchor represents neutral posture (subject follows protocol)
  - Eliminates dependence on L/R shoulder-rotation assumption
  - Fix doesn't require new recordings — works on all existing data

Anchor formula:
  anchor = mean(first_k sit_straight frames)   [current: k=25 from L+R]
                                                 [new:     k=25 from sit]

Variants:
  E2a1: sit_anchor + E1i offset architecture (main method, apples-to-apples vs L/R anchor)
  E2a2: sit_anchor + E1f concat architecture (comparison)

Usage:
  python3 scripts/ms_204_sit_anchor.py --seed 42
"""
import sys, json, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import warnings; warnings.filterwarnings("ignore")

from ms_140_common import CLASSES_4, WINDOW_T
from ms_62_day2_targeted import compute_subject_bone_reference, compute_node_features_hybrid
from ms_73_day2plus_ergo import HIDDEN, TemporalSkeletonBranch
from ms_114_ecgf_xs import set_seed
from ms_173_final_9subj import (
    SUBJECTS_9, NEW_SUBJECTS, load_nom_segments, compute_features,
    FocalLoss, evaluate_model,
    TRAIN_STRIDE, EVAL_STRIDE, EPOCHS, BATCH, DEVICE, OUT_DIR,
    Dataset9Subj,
)
from ms_176_offset_arch import SkelWithOffset


def compute_sit_anchors(segments, n=25):
    """Per-subject feature anchors from sit_straight NOM samples."""
    anchors = {}
    for subj in SUBJECTS_9:
        key = (subj, "sit_straight", "nom")
        if key not in segments: continue
        feats_list = []
        for f in segments[key][:n]:
            w = np.load(f["lm_world"]).astype(np.float32)
            img = np.load(f["lm_img"]).astype(np.float32)
            dp = None
            if f["depthpro"] and Path(f["depthpro"]).exists():
                dp = np.load(f["depthpro"]).astype(np.float32)
                if dp.shape != (224, 224): dp = cv2.resize(dp, (224, 224))
            feats_list.append(compute_features(w, img, dp))
        if feats_list:
            anchors[subj] = np.mean(feats_list, axis=0).astype(np.float32)
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    print("Loading 9 subjects (NOM only)...", flush=True)
    all_segs, rows = load_nom_segments()
    brefs = {s: compute_subject_bone_reference(rows[s]) for s in SUBJECTS_9 if s in rows}
    sit_anchors = compute_sit_anchors(all_segs, n=25)

    print("\nSIT-based anchor values per subject:", flush=True)
    for s in SUBJECTS_9:
        if s in sit_anchors:
            a = sit_anchors[s]
            print(f"  {s:>8}: sh/ear={a[0]:.3f}  neck={a[1]:.3f}  HFD={a[2]:+.4f}  CVA={a[3]:+.2f}", flush=True)

    fold_results = []
    t0 = time.time()

    for fi, held_out in enumerate(SUBJECTS_9):
        if held_out not in brefs: continue
        train_pool = [s for s in SUBJECTS_9 if s != held_out and s in brefs]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]

        train_seg = {k: v for k, v in all_segs.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in all_segs.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in all_segs.items() if k[0] == held_out}

        # Use sit_anchors instead of lr_anchors
        train_ds = Dataset9Subj(train_seg, brefs, sit_anchors,
                                 stride=TRAIN_STRIDE, augment=True,
                                 use_anchor=True, feature_mode='full')
        val_ds = Dataset9Subj(val_seg, brefs, sit_anchors,
                               stride=EVAL_STRIDE, augment=False,
                               use_anchor=True, feature_mode='full')
        test_ds = Dataset9Subj(test_seg, brefs, sit_anchors,
                                stride=EVAL_STRIDE, augment=False,
                                use_anchor=True, feature_mode='full')

        print(f"\n=== fold {fi+1}/9: held_out={held_out} (val={val_subj}) ===", flush=True)

        # Train E1i with sit anchor
        set_seed(args.seed + fi)
        g = torch.Generator(); g.manual_seed(args.seed + fi)
        train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True, generator=g)
        model = SkelWithOffset(3).to(DEVICE)
        param_groups = [{"params": m.parameters(), "lr": 5e-4 if n == "skeleton" else 1e-3}
                        for n, m in model.named_children() if list(m.parameters())]
        opt = torch.optim.AdamW(param_groups, weight_decay=0.02)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = FocalLoss(gamma=2.0)
        best_val, best_state = -1.0, None
        for ep in range(1, EPOCHS + 1):
            model.train()
            for batch in train_ld:
                bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(DEVICE)
                loss = crit(model(bd), labels)
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
        test_res = evaluate_model(model, test_ds)

        pc = test_res["per_class"]
        minv = min(pc.values())
        both = "BOTH>=90" if pc['forward_head'] >= 90 and pc['sit_straight'] >= 90 else ""
        all4 = "ALL4>=90" if minv >= 90 else ""
        no_zero = "NO_ZERO" if minv >= 70 else f"MIN={minv:.0f}"
        tag = "(NEW)" if held_out in NEW_SUBJECTS else ""
        marker = " <-- PAN" if held_out == "pan" else (" <-- MUK" if held_out == "mukrop" else "")
        print(f"  [{held_out}] {test_res['macro']:.1f}%  fh={pc['forward_head']:.0f} sit={pc['sit_straight']:.0f} "
              f"L={pc['left_leaning']:.0f} R={pc['right_leaning']:.0f}  "
              f"{both} {all4} {no_zero} {tag}{marker}", flush=True)
        fold_results.append({"held_out": held_out, "test": test_res})

    # Summary
    m_all = [f["test"]["macro"] for f in fold_results]
    m_8 = [f["test"]["macro"] for f in fold_results if f["held_out"] != "pan"]
    no_zero = sum(1 for f in fold_results if all(f["test"]["per_class"][c] >= 70 for c in CLASSES_4))
    all4 = sum(1 for f in fold_results if all(f["test"]["per_class"][c] >= 90 for c in CLASSES_4))
    pan_f = [f for f in fold_results if f["held_out"] == "pan"]
    pan_pc = pan_f[0]["test"]["per_class"] if pan_f else {}
    pan_ok = pan_f and all(v >= 70 for v in pan_pc.values())

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"E2a1 (SIT-anchor + E1i offset) RESULTS ({elapsed:.0f} min)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  9-subj macro: {np.mean(m_all):.1f}%  8-subj (no pan): {np.mean(m_8):.1f}%", flush=True)
    print(f"  no_class<70: {no_zero}/9  all4>=90: {all4}/9", flush=True)
    if pan_pc:
        print(f"  pan: fh={pan_pc['forward_head']:.0f} sit={pan_pc['sit_straight']:.0f} "
              f"L={pan_pc['left_leaning']:.0f} R={pan_pc['right_leaning']:.0f}  pan_all>=70: {pan_ok}", flush=True)

    with open(OUT_DIR / "ms204_sit_anchor.json", "w") as f:
        json.dump({"E2a1": {"folds": fold_results,
                             "mean_9": float(np.mean(m_all)),
                             "mean_8": float(np.mean(m_8)),
                             "no_zero_9": no_zero,
                             "all4_9": all4,
                             "pan_ok": bool(pan_ok),
                             "desc": "SIT-anchor + E1i offset (protocol: neutral sit as calibration)"}},
                   f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'ms204_sit_anchor.json'}", flush=True)


if __name__ == "__main__":
    main()
