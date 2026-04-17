"""
ms_114 — ECGF-XS: ECGF + side-view fh_distance auxiliary regression head.

NOVEL CONTRIBUTION
==================
Starting from the Day 13 ECGF architecture, we add:

  (1) **fh_distance auxiliary regression head** — a second regression head
      (alongside the existing RULA head) that predicts the side-view forward
      head distance (ear-to-shoulder horizontal offset, normalized by
      shoulder-ear distance). Ground truth from `side_supervision.csv`.
      Privileged training: side-view available only at train time.

  (2) **16-dim ergonomic features** including the DA3 head−shoulder depth
      differential (`compute_ergonomic_window_with_depth`). Relative
      scale-invariant depth feature. Deployable from a single front cam.

  (3) **Class-balanced focal loss** with heavy weighting on forward_head and
      slouched_posture (the two historically weak classes).

At inference time ONLY the front camera is used — the fh_distance head's
output is discarded, but the encoder has learned a representation that
internalizes side-view depth understanding during training.

Target: ≥80% per-class accuracy on every class (5-fold LOSO).

Usage:
  python3 scripts/ms_114_ecgf_xs.py --tag xs_v1

Output:
  checkpoints/ecgf_xs_<tag>/
  outputs/ecgf_xs_<tag>_results.{json,md}
"""

import sys, csv, time, json, warnings, argparse, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ms_62_day2_targeted import (
    compute_subject_bone_reference, compute_node_features_hybrid, FocalLoss
)
from ms_73_day2plus_ergo import (
    SUBJECTS, CLASSES, WINDOW_T, TRAIN_STRIDE, EVAL_STRIDE,
    EPOCHS, WARMUP, BATCH, HIDDEN, DROPOUT,
    load_segments, compute_depth_stats,
)
from ms_71c_ergonomic_features import (
    compute_ergonomic_window, compute_ergonomic_window_with_depth,
    ErgonomicBranch,
)
from ms_82_train_synth_augmented import load_synth_segments
from ms_84_ecgf import (
    TemporalTeacherECGF, LAMBDA_RULA, evaluate,
)

OUT_DIR = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE_CSV = ROOT / "data" / "multisubject" / "side_supervision.csv"

# Heavier focal weight on the two hard classes
CLASS_WEIGHTS_XS = torch.tensor([5.0, 1.0, 1.0, 1.0, 3.0], dtype=torch.float32)
FOCAL_GAMMA_XS = 3.0
LAMBDA_FH = 0.3           # fh regression weight
LAMBDA_RULA_XS = 0.05     # keep RULA aux

EXCLUDE_XS = [
    ("mukrop", "sit_straight", "close"),
    ("boom", "forward_head", "far"),
    ("boom", "forward_head", "nom"),
    ("fu", "forward_head", "far"),
    ("mukrop", "right_leaning", "nom"),
    ("nonny", "left_leaning", "far"),
    ("peemai", "forward_head", "nom"),
    ("fu", "sit_straight", "nom"),  # corrupted — irrecoverable
]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_side_supervision():
    """Returns dict: (subject, class, distance, frame_stem) -> fh_distance scalar."""
    sup = {}
    if not SIDE_CSV.exists():
        print(f"  WARNING: {SIDE_CSV} not found — fh supervision disabled")
        return sup, 0.0, 1.0
    with open(SIDE_CSV) as f:
        for r in csv.DictReader(f):
            if int(r["valid"]) == 0:
                continue
            key = (r["subject"], r["class"], r["distance"], r["frame_stem"])
            sup[key] = float(r["fh_distance"])
    vals = np.array(list(sup.values()))
    mean, std = float(vals.mean()), float(vals.std() + 1e-6)
    print(f"  Loaded {len(sup)} side supervision entries. mean={mean:.4f} std={std:.4f}")
    return sup, mean, std


# ── Dataset ────────────────────────────────────────────────────────────────

class WindowedDatasetXS(torch.utils.data.Dataset):
    """
    Extension of WindowedDatasetErgo that:
      - Stores image landmarks + stem + (subj, class, dist) per frame
      - Computes 16-dim ergo features (with DA3 head-shoulder depth diff)
      - Looks up side-view fh_distance per window (uses middle frame's stem)
      - Returns a `has_sup` mask for windows without side supervision
    """
    def __init__(self, segment_dict, bone_refs, side_sup, sup_mean, sup_std,
                 depth_stats=None, window_t=WINDOW_T, stride=TRAIN_STRIDE,
                 augment=False, modality_dropout=0.0):
        self.bone_refs = bone_refs
        self.window_t = window_t
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._depth_stats = depth_stats
        self.side_sup = side_sup
        self.sup_mean = sup_mean
        self.sup_std = sup_std
        self.segments = []

        for key, frames in segment_dict.items():
            subj, cls_name, dist = key[0], key[1], key[2]
            label = frames[0]["label"]
            ref = bone_refs[subj]
            seg_frames = []
            for f in frames:
                world = np.load(f["lm_world"]).astype(np.float32)
                img_lm = np.load(f["lm_img"]).astype(np.float32)
                node = compute_node_features_hybrid(world, img_lm, ref)

                if f["da3_depth"] and Path(f["da3_depth"]).exists():
                    da3 = np.load(f["da3_depth"]).astype(np.float32)
                    if da3.shape != (224, 224):
                        da3 = cv2.resize(da3, (224, 224), interpolation=cv2.INTER_LINEAR)
                else:
                    da3 = np.zeros((224, 224), dtype=np.float32)
                has_rs = False
                if f.get("rs_depth") and Path(f["rs_depth"]).exists():
                    rs_full = np.load(f["rs_depth"]).astype(np.float32)
                    rs = cv2.resize(rs_full, (224, 224), interpolation=cv2.INTER_LINEAR)
                    has_rs = True
                else:
                    rs = np.zeros((224, 224), dtype=np.float32)
                if f.get("dinov3"):
                    dino = np.load(f["dinov3"]).astype(np.float32)
                else:
                    dino = np.zeros(384, dtype=np.float32)

                # stem for side supervision lookup (strip any _dtN synth suffix)
                stem_full = f.get("frame", "0000")
                stem = stem_full.split("_")[0] if isinstance(stem_full, str) else str(stem_full)
                seg_frames.append({
                    "node": node, "world": world, "img": img_lm,
                    "_da3": da3, "_rs": rs, "has_rs": has_rs, "dino": dino,
                    "subj": subj, "cls": cls_name, "dist": dist, "stem": stem,
                })
            self.segments.append({"subject": subj, "label": label, "frames": seg_frames,
                                   "cls_name": cls_name, "dist": dist})

        self.windows = []
        for si, seg in enumerate(self.segments):
            n = len(seg["frames"])
            if n < window_t: continue
            for start in range(0, n - window_t + 1, stride):
                self.windows.append((si, start))

    def set_depth_stats(self, stats): self._depth_stats = stats

    def collect_depth_for_stats(self, n_samples=400):
        da3, rs = [], []
        idx = np.random.permutation(len(self.segments))
        c = 0
        for si in idx:
            for f in self.segments[si]["frames"]:
                da3.append(f["_da3"].flatten())
                if f["has_rs"]: rs.append(f["_rs"].flatten())
                c += 1
                if c >= n_samples: break
            if c >= n_samples: break
        return np.concatenate(da3), np.concatenate(rs) if rs else np.array([0.0])

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        si, start = self.windows[idx]
        seg = self.segments[si]
        frames = seg["frames"][start:start + self.window_t]

        node = np.stack([f["node"] for f in frames], axis=0)
        if self.augment:
            node = node + np.random.randn(*node.shape).astype(np.float32) * 0.005
            if np.random.rand() < 0.5:
                sx = np.random.uniform(0.92, 1.08)
                sy = np.random.uniform(0.92, 1.08)
                node[..., 0] *= sx; node[..., 1] *= sy
                node[..., 26] *= sx; node[..., 27] *= sy

        world_t = np.stack([f["world"] for f in frames], axis=0)
        img_t = np.stack([f["img"] for f in frames], axis=0)
        da3_t_mid = frames[self.window_t // 2]["_da3"]

        # 16-dim ergo with depth differential
        ergo = compute_ergonomic_window_with_depth(world_t, da3_t_mid, img_t)

        mid = self.window_t // 2
        mf = frames[mid]
        da3 = mf["_da3"]; rs = mf["_rs"]; has_rs = mf["has_rs"]
        s = self._depth_stats
        if s is not None:
            da3_n = (da3 - s["da3_mean"]) / s["da3_std"]
            rs_n = (rs - s["rs_mean"]) / s["rs_std"]
        else:
            da3_n = da3; rs_n = rs
        if not has_rs: rs_n = np.zeros_like(rs_n)
        depth_2ch = np.stack([da3_n, rs_n], axis=0).astype(np.float32)

        dino = np.mean([f["dino"] for f in frames], axis=0).astype(np.float32)

        if self.augment and self.modality_dropout > 0:
            r = np.random.rand()
            if r < self.modality_dropout / 2:
                depth_2ch = np.zeros_like(depth_2ch)
            elif r < self.modality_dropout:
                dino = np.zeros_like(dino)

        # Side supervision lookup using middle frame's (subj, cls, dist, stem)
        sup_key = (mf["subj"], mf["cls"], mf["dist"], mf["stem"])
        fh_raw = self.side_sup.get(sup_key, None)
        if fh_raw is not None:
            fh_norm = (fh_raw - self.sup_mean) / self.sup_std
            has_sup = 1.0
        else:
            fh_norm = 0.0; has_sup = 0.0

        return {
            "node": torch.tensor(node, dtype=torch.float32),
            "depth": torch.tensor(depth_2ch, dtype=torch.float32),
            "dino": torch.tensor(dino, dtype=torch.float32),
            "ergo": torch.tensor(ergo, dtype=torch.float32),
            "fh_dist": torch.tensor(fh_norm, dtype=torch.float32),
            "has_sup": torch.tensor(has_sup, dtype=torch.float32),
            "label": seg["label"],
        }


# ── Model: ECGF + fh_head ─────────────────────────────────────────────────

class TemporalTeacherECGF_XS(TemporalTeacherECGF):
    """ECGF with 16-dim ergo (depth_diff) + fh_distance auxiliary head."""

    def __init__(self, hidden=HIDDEN, n_classes=5, dropout=DROPOUT):
        # Parent constructor already wires ergo/fusion to the requested dim
        super().__init__(hidden=hidden, n_classes=n_classes, dropout=dropout, ergo_dim=16)
        # fh head takes the same fused (hidden * 5) representation as cls_head
        self.fh_head = nn.Sequential(
            nn.Linear(hidden * 5, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch, return_aux=False):
        sk = self.skeleton(batch["node"])
        dp = self.depth(batch["depth"])
        di = self.dino(batch["dino"])
        er = self.ergo(batch["ergo"])
        fused, gate_w = self.fusion([sk, dp, di, er], batch["ergo"])
        cls_logits = self.cls_head(fused)
        rula_pred = self.rula_head(fused).squeeze(-1)
        fh_pred = self.fh_head(fused).squeeze(-1)
        if return_aux:
            return cls_logits, rula_pred, fh_pred, gate_w
        return cls_logits


# ── Training ───────────────────────────────────────────────────────────────

def train_fold(train_seg, val_seg, test_seg, bone_refs, side_sup, sup_mean, sup_std,
               fold_name, seed):
    set_seed(seed)
    train_ds = WindowedDatasetXS(train_seg, bone_refs, side_sup, sup_mean, sup_std,
                                  stride=TRAIN_STRIDE, augment=True, modality_dropout=0.1)
    val_ds = WindowedDatasetXS(val_seg, bone_refs, side_sup, sup_mean, sup_std,
                                stride=EVAL_STRIDE, augment=False, modality_dropout=0.0)
    test_ds = WindowedDatasetXS(test_seg, bone_refs, side_sup, sup_mean, sup_std,
                                 stride=EVAL_STRIDE, augment=False, modality_dropout=0.0)
    stats = compute_depth_stats(train_ds)
    for ds in (train_ds, val_ds, test_ds): ds.set_depth_stats(stats)
    g = torch.Generator(); g.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)

    model = TemporalTeacherECGF_XS(hidden=HIDDEN, n_classes=5, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": model.skeleton.parameters(), "lr": 5e-4},
        {"params": model.depth.parameters(), "lr": 1e-3},
        {"params": model.dino.parameters(), "lr": 1e-3},
        {"params": model.ergo.parameters(), "lr": 1e-3},
        {"params": model.fusion.parameters(), "lr": 1e-3},
        {"params": model.cls_head.parameters(), "lr": 1e-3},
        {"params": model.rula_head.parameters(), "lr": 1e-3},
        {"params": model.fh_head.parameters(), "lr": 1e-3},
    ], weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    cls_crit = FocalLoss(gamma=FOCAL_GAMMA_XS, weight=CLASS_WEIGHTS_XS.to(DEVICE))

    best_val, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        if ep <= WARMUP:
            for p in model.depth.parameters(): p.requires_grad_(False)
            for p in model.dino.parameters(): p.requires_grad_(False)
        elif ep == WARMUP + 1:
            for p in model.depth.parameters(): p.requires_grad_(True)
            for p in model.dino.parameters(): p.requires_grad_(True)

        for batch in train_ld:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k not in ("label",)}
            labels = batch["label"].to(DEVICE) if isinstance(batch["label"], torch.Tensor) \
                else torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)
            cls_logits, rula_pred, fh_pred, _ = model(bd, return_aux=True)
            rula_target = bd["ergo"][:, 14]
            fh_target = bd["fh_dist"]
            has_sup = bd["has_sup"]
            l_cls = cls_crit(cls_logits, labels)
            l_rula = F.mse_loss(rula_pred, rula_target)
            # Masked fh regression
            fh_sq = (fh_pred - fh_target) ** 2
            l_fh = (fh_sq * has_sup).sum() / (has_sup.sum() + 1e-6)
            loss = l_cls + LAMBDA_RULA_XS * l_rula + LAMBDA_FH * l_fh
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        val_res = evaluate(model, val_ds)
        if val_res["macro"] > best_val:
            best_val = val_res["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  [{fold_name}] ep {ep:3d}/{EPOCHS} val={val_res['macro']:.2f}% best={best_val:.2f}%", flush=True)
    model.load_state_dict(best_state); model.eval()
    return best_state, best_val, evaluate(model, test_ds), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xs_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    CKPT_DIR = ROOT / "checkpoints" / f"ecgf_xs_{args.tag}"
    CKPT_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print(f"ms_114 ECGF-XS  tag={args.tag}  seed={args.seed}")
    print(f"  ergo: 16-dim (15 RULA + 1 DA3 head-shoulder diff)")
    print(f"  aux losses: RULA (lambda={LAMBDA_RULA_XS})  fh_distance (lambda={LAMBDA_FH})")
    print(f"  focal gamma={FOCAL_GAMMA_XS}  class_weights={CLASS_WEIGHTS_XS.tolist()}")
    print("=" * 70)

    side_sup, sup_mean, sup_std = load_side_supervision()
    real_segments, rows_per_subj = load_segments(exclude_keys=EXCLUDE_XS)
    synth_segments = load_synth_segments(real_segments)
    bad_synth = ("fu", "slouched_posture", "nom_synth")
    if bad_synth in synth_segments:
        del synth_segments[bad_synth]
    combined = dict(real_segments); combined.update(synth_segments)
    bone_refs = {s: compute_subject_bone_reference(rows_per_subj[s]) for s in SUBJECTS}
    print(f"Loaded {len(combined)} segments (real+synth)")

    fold_results = []
    t0 = time.time()
    for fi, held_out in enumerate(SUBJECTS):
        train_pool = [s for s in SUBJECTS if s != held_out]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]
        print(f"\nFOLD {fi+1}/5  held_out={held_out}  val={val_subj}  train={train_subjects}  ({(time.time()-t0)/60:.1f} min)")
        train_seg = {k: v for k, v in combined.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in real_segments.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in real_segments.items() if k[0] == held_out}

        best_state, best_val, test_res, stats = train_fold(
            train_seg, val_seg, test_seg, bone_refs, side_sup, sup_mean, sup_std,
            f"fold{fi+1}_{held_out}", args.seed + fi
        )
        torch.save({"model_state": best_state, "test": test_res, "val_macro": best_val,
                    "depth_stats": stats, "held_out": held_out,
                    "bone_refs": bone_refs, "sup_mean": sup_mean, "sup_std": sup_std,
                    "seed": args.seed + fi}, CKPT_DIR / f"{held_out}_best.pt")
        print(f"  [{held_out}] val={best_val:.2f}% test={test_res['macro']:.2f}% per_class={test_res['per_class']}", flush=True)
        fold_results.append({
            "fold": fi+1, "held_out": held_out, "val_subject": val_subj,
            "val_macro": best_val, "test": test_res,
        })
        with open(OUT_DIR / f"ecgf_xs_{args.tag}_results.json", "w") as f:
            json.dump({"tag": args.tag, "seed": args.seed, "folds": fold_results,
                       "class_weights": CLASS_WEIGHTS_XS.tolist(),
                       "focal_gamma": FOCAL_GAMMA_XS,
                       "lambda_fh": LAMBDA_FH, "lambda_rula": LAMBDA_RULA_XS}, f, indent=2)

    macros = [f["test"]["macro"] for f in fold_results]
    per_class_means = {c: float(np.mean([f["test"]["per_class"][c] for f in fold_results])) for c in CLASSES}
    above_80 = sum(1 for v in per_class_means.values() if v >= 80)
    print(f"\nDONE  tag={args.tag}  Mean: {np.mean(macros):.2f} +/- {np.std(macros):.2f}")
    print(f"  Per-class mean: {per_class_means}")
    print(f"  Classes >= 80: {above_80}/5")

    md = [f"# ECGF-XS ({args.tag})  — Day 13 + fh regression + 16-dim ergo + heavy focal\n",
          f"seed={args.seed}  lambda_fh={LAMBDA_FH}  lambda_rula={LAMBDA_RULA_XS}",
          f"focal_gamma={FOCAL_GAMMA_XS}  class_weights={CLASS_WEIGHTS_XS.tolist()}\n",
          "## Per-fold\n",
          "| held_out | val | test_macro | fh | left | right | sit | sl |",
          "|---|---|---|---|---|---|---|---|"]
    for f in fold_results:
        pc = f["test"]["per_class"]
        md.append(f"| {f['held_out']} | {f['val_macro']:.1f} | **{f['test']['macro']:.1f}** | "
                  f"{pc['forward_head']:.0f} | {pc['left_leaning']:.0f} | "
                  f"{pc['right_leaning']:.0f} | {pc['sit_straight']:.0f} | {pc['slouched_posture']:.0f} |")
    md.append(f"| **mean +/- std** | | **{np.mean(macros):.2f} +/- {np.std(macros):.2f}** | | | | | |")
    md.append("\n## Per-class mean (every class must be >=80)\n")
    md.append("| class | mean | std | min | max | >=80? |")
    md.append("|---|---|---|---|---|---|")
    for c in CLASSES:
        vs = [f["test"]["per_class"][c] for f in fold_results]
        ok = "YES" if np.mean(vs) >= 80 else "NO"
        md.append(f"| {c} | {np.mean(vs):.1f} | {np.std(vs):.1f} | {min(vs):.0f} | {max(vs):.0f} | {ok} |")
    (OUT_DIR / f"ecgf_xs_{args.tag}_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Report: {OUT_DIR / f'ecgf_xs_{args.tag}_results.md'}")


if __name__ == "__main__":
    main()
