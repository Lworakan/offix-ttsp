"""
ms_176 — E1h: Offset Architecture — extra features ONLY modify fh/sit logits.

Problem: concat([skel, extra]) → shared cls_head lets extra features
corrupt L/R decisions. E1c got 92.8% but mukrop L dropped 100→63.

Fix: Extra features add an OFFSET to fh/sit logits only.
L/R logits are purely from skeleton — physically IMPOSSIBLE to corrupt.

E1h: skel cls_head(128→4) + offset(HFD+CVA→fh,sit) — L/R protected
E1i: same but with sh/ear+HFD+CVA

Usage:
  python3 scripts/ms_176_offset_arch.py --seed 42
"""
import sys, json, time, argparse
from pathlib import Path
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
    compute_lr_anchors, Dataset9Subj, FocalLoss, evaluate_model,
    TRAIN_STRIDE, EVAL_STRIDE, EPOCHS, BATCH, DEVICE, OUT_DIR,
)


class SkelWithOffset(nn.Module):
    """Skeleton decides all 4 classes. Extra features ONLY offset fh/sit logits."""
    def __init__(self, extra_dim, h=HIDDEN, nc=4, do=0.15):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        # Skeleton-only classifier for ALL 4 classes
        self.cls_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
        # fh/sit offset from extra features — CANNOT affect L/R
        if extra_dim > 0:
            self.fh_offset = nn.Sequential(nn.Linear(extra_dim, 16), nn.GELU(), nn.Linear(16, 1))
            self.sit_offset = nn.Sequential(nn.Linear(extra_dim, 16), nn.GELU(), nn.Linear(16, 1))
        else:
            self.fh_offset = None
            self.sit_offset = None

    def forward(self, batch, **kw):
        sk = self.skeleton(batch["node"])
        logits = self.cls_head(sk)  # (B, 4) — all from skeleton
        if self.fh_offset is not None and batch["extra"].shape[1] > 0:
            extra = batch["extra"]
            # ONLY modify forward_head (index 0) and sit_straight (index 3)
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + self.fh_offset(extra).squeeze(-1)
            logits[:, 3] = logits[:, 3] + self.sit_offset(extra).squeeze(-1)
            # indices 1 (left) and 2 (right) are UNTOUCHED
        return logits


def train_fold(model_cls_fn, train_ds, val_ds, test_ds, seed):
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)
    model = model_cls_fn().to(DEVICE)
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
    return evaluate_model(model, test_ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    print("Loading 9 subjects (NOM only)...", flush=True)
    all_segs, rows = load_nom_segments()
    brefs = {s: compute_subject_bone_reference(rows[s]) for s in SUBJECTS_9 if s in rows}
    anchors = compute_lr_anchors(all_segs)

    EXPERIMENTS = {
        'E1h': {'mode': 'depth', 'extra_dim': 2, 'desc': 'skel + OFFSET anchored HFD+CVA (L/R protected)'},
        'E1i': {'mode': 'full',  'extra_dim': 3, 'desc': 'skel + OFFSET anchored sh/ear+HFD+CVA (L/R protected)'},
    }

    all_results = {}
    t0 = time.time()

    for ename, ecfg in EXPERIMENTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"{ename}: {ecfg['desc']}", flush=True)
        print(f"{'='*60}", flush=True)

        fold_results = []
        for fi, held_out in enumerate(SUBJECTS_9):
            if held_out not in brefs: continue
            train_pool = [s for s in SUBJECTS_9 if s != held_out and s in brefs]
            val_subj = train_pool[fi % len(train_pool)]
            train_subjects = [s for s in train_pool if s != val_subj]

            train_seg = {k: v for k, v in all_segs.items() if k[0] in train_subjects}
            val_seg = {k: v for k, v in all_segs.items() if k[0] == val_subj}
            test_seg = {k: v for k, v in all_segs.items() if k[0] == held_out}

            train_ds = Dataset9Subj(train_seg, brefs, anchors, stride=TRAIN_STRIDE,
                                    augment=True, use_anchor=True, feature_mode=ecfg['mode'])
            val_ds = Dataset9Subj(val_seg, brefs, anchors, stride=EVAL_STRIDE,
                                  augment=False, use_anchor=True, feature_mode=ecfg['mode'])
            test_ds = Dataset9Subj(test_seg, brefs, anchors, stride=EVAL_STRIDE,
                                   augment=False, use_anchor=True, feature_mode=ecfg['mode'])

            model_fn = lambda ed=ecfg['extra_dim']: SkelWithOffset(ed)
            test_res = train_fold(model_fn, train_ds, val_ds, test_ds, args.seed + fi)
            pc = test_res["per_class"]
            fh = pc["forward_head"]; sit = pc["sit_straight"]
            both = "BOTH>=90" if fh >= 90 and sit >= 90 else ""
            all4 = "ALL4>=90" if all(v >= 90 for v in pc.values()) else ""
            tag = "(NEW)" if held_out in NEW_SUBJECTS else ""
            print(f"  [{held_out}] {test_res['macro']:.1f}%  fh={fh:.0f} sit={sit:.0f} "
                  f"L={pc['left_leaning']:.0f} R={pc['right_leaning']:.0f}  {both} {all4} {tag}", flush=True)
            fold_results.append({"held_out": held_out, "test": test_res})

        all_f = [f for f in fold_results if f["held_out"] != "pan"]
        m = [f["test"]["macro"] for f in all_f]
        both_count = sum(1 for f in all_f if f["test"]["per_class"]["forward_head"] >= 90
                         and f["test"]["per_class"]["sit_straight"] >= 90)
        all4_count = sum(1 for f in all_f if all(f["test"]["per_class"][c] >= 90
                         for c in CLASSES_4))
        print(f"\n  [{ename}] 8-subj: {np.mean(m):.1f}%  both={both_count}/8  all4={all4_count}/8", flush=True)
        all_results[ename] = {"folds": fold_results, "mean_8": float(np.mean(m)),
                               "both_90": both_count, "all4_90": all4_count, "desc": ecfg['desc']}

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"OFFSET ARCHITECTURE RESULTS ({elapsed:.0f} min)", flush=True)
    print(f"{'='*70}", flush=True)
    for e, r in all_results.items():
        print(f"  {e}: {r['desc']}", flush=True)
        print(f"      {r['mean_8']:.1f}%  both={r['both_90']}/8  all4={r['all4_90']}/8", flush=True)

    # Compare with E1a and E1c
    q1_path = OUT_DIR / "ms173_final.json"
    if q1_path.exists():
        q1 = json.load(open(q1_path))
        print(f"\n  COMPARISON:", flush=True)
        print(f"  E1a (skel only):     {q1['E1a']['mean_8']:.1f}%  all4={q1['E1a']['all4_90']}/8", flush=True)
        print(f"  E1c (skel+HFD+CVA):  {q1['E1c']['mean_8']:.1f}%  all4={q1['E1c']['all4_90']}/8", flush=True)
        for e, r in all_results.items():
            print(f"  {e} (offset):        {r['mean_8']:.1f}%  all4={r['all4_90']}/8", flush=True)

    with open(OUT_DIR / "ms176_offset.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'ms176_offset.json'}", flush=True)


if __name__ == "__main__":
    main()
