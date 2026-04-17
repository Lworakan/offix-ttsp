"""
Day 7 — Day 2 architecture + ergonomic features, from scratch, on our 5 subjects.

Combines:
  ✓ Day 2's hybrid 52-feat skeleton (raw 26 + bone-normalized 26)
  ✓ DA3 + RealSense depth (2-channel)
  ✓ DINOv3 features (384-d)
  ✓ NEW: Ergonomic features (15-d, RULA/REBA/CVA grounded)
  ✓ Day 2's hyperparameters: 120 epochs, focal loss γ=2 fh-weight=3,
    body-shape augmentation, 4-modality gated fusion
  ✗ No POLAR pretraining (it didn't transfer)

Run via:
  tmux new -s day7
  python3 scripts/ms_73_day2plus_ergo.py 2>&1 | tee /tmp/day7.log
  Ctrl+B  D

Optional cleaned-dataset run:
  python3 scripts/ms_73_day2plus_ergo.py --exclude_segments mukrop/sit_straight/close --out_suffix _clean
"""

import sys, csv, time, json, warnings, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from ms_62_day2_targeted import (
    BODY_EDGES, ADJ_LIST, REF_BONES, GCNLayer, FocalLoss,
    DepthBranch, DINOBranch, GatedFusion,
    compute_subject_bone_reference, compute_node_features_hybrid,
    DROP_SUBJECTS, EXCLUDE_CLASSES,
)
from ms_71c_ergonomic_features import compute_ergonomic_window, ErgonomicBranch

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "multisubject" / "webcam_dataset.csv"
DINOV3_ROOT = ROOT / "data" / "multisubject" / "dinov3_features"
CKPT_DIR = ROOT / "checkpoints" / "day7_day2plus_ergo"
OUT_DIR = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUBJECTS = ["fu", "mukrop", "nonny", "boom", "peemai"]
CLASSES = ["forward_head", "left_leaning", "right_leaning", "sit_straight", "slouched_posture"]

# Day 2 hyperparameters (the recipe that worked)
WINDOW_T = 7
TRAIN_STRIDE = 2
EVAL_STRIDE = 7
EPOCHS = 120
WARMUP = 15
BATCH = 32
HIDDEN = 128
DROPOUT = 0.15
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.5], dtype=torch.float32)


# ── Model: Day 2 teacher + ergonomic branch ──────────────────────────────

class TemporalSkeletonBranch(nn.Module):
    def __init__(self, n_feat=52, hidden=128, n_layers=3, dropout=0.15):
        super().__init__()
        adj = torch.eye(33)
        for i, j in BODY_EDGES: adj[i, j] = 1.0; adj[j, i] = 1.0
        deg = adj.sum(1).clamp(min=1)
        adj_norm = deg.pow(-0.5).unsqueeze(1) * adj * deg.pow(-0.5).unsqueeze(0)
        self.register_buffer("adj", adj_norm)
        layers = [GCNLayer(n_feat, hidden, self.adj)]
        for _ in range(n_layers - 1):
            layers.append(GCNLayer(hidden, hidden, self.adj))
        self.gcn = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.t_attn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        B, T, J, F_ = x.shape
        x = x.reshape(B * T, J, F_)
        for layer in self.gcn:
            x = layer(x); x = self.drop(x)
        x = x.mean(dim=1).reshape(B, T, -1)
        attn = torch.softmax(self.t_attn(x), dim=1)
        return (x * attn).sum(dim=1)


class TemporalTeacherErgo(nn.Module):
    """Day 2 teacher + ergonomic branch (4 modalities, 5 classes)."""
    def __init__(self, hidden=128, n_classes=5, dropout=0.15):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=hidden, dropout=dropout)
        self.depth = DepthBranch(hidden=hidden)
        self.dino = DINOBranch(in_dim=384, hidden=hidden, dropout=dropout)
        self.ergo = ErgonomicBranch(in_dim=15, hidden=hidden, dropout=dropout)
        self.fusion = GatedFusion(4, hidden)
        # Fusion outputs hidden + 4*hidden = 5*hidden
        self.head = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch):
        sk = self.skeleton(batch["node"])
        dp = self.depth(batch["depth"])
        di = self.dino(batch["dino"])
        er = self.ergo(batch["ergo"])
        fused = self.fusion([sk, dp, di, er])
        return self.head(fused)


# ── Dataset: Day 2 hybrid 52-feat + ergo per window ──────────────────────

def load_segments(exclude_keys=None):
    exclude_keys = set(exclude_keys or [])
    segments = defaultdict(list)
    rows_per_subj = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["subject"] in DROP_SUBJECTS: continue
            if r["class"] in EXCLUDE_CLASSES: continue
            if r["class"] not in CLASSES: continue
            key = (r["subject"], r["class"], r.get("distance", "nom"))
            if key in exclude_keys: continue
            lm = r.get("webcam_landmarks", "")
            if not lm or not Path(lm).exists(): continue
            stem = r.get("frame_stem", "0")
            dino_path = DINOV3_ROOT / r["subject"] / r["class"] / r.get("distance", "nom") / f"{stem}_dino.npy"
            segments[key].append({
                "lm_world": lm,
                "lm_img": lm.replace("_landmarks.npy", "_landmarks_img.npy"),
                "da3_depth": r.get("webcam_da3_depth", "") or None,
                "rs_depth": r.get("rs_depth_raw", "") or None,
                "dinov3": str(dino_path) if dino_path.exists() else None,
                "label": CLASSES.index(r["class"]),
                "subject": r["subject"], "frame": stem,
            })
            rows_per_subj[r["subject"]].append(r)
    for k in segments:
        segments[k].sort(key=lambda f: int(f["frame"]))
    return segments, rows_per_subj


class WindowedDatasetErgo(Dataset):
    def __init__(self, segments, bone_refs, depth_stats=None,
                 window_t=WINDOW_T, stride=TRAIN_STRIDE,
                 augment=False, modality_dropout=0.0):
        self.bone_refs = bone_refs
        self.window_t = window_t
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._depth_stats = depth_stats
        self.segments = []

        for key, frames in segments.items():
            subj = key[0]
            label = frames[0]["label"]
            ref = bone_refs[subj]
            seg_frames = []
            for f in frames:
                world = np.load(f["lm_world"]).astype(np.float32)
                img_lm = np.load(f["lm_img"]).astype(np.float32)
                node = compute_node_features_hybrid(world, img_lm, ref)  # (33, 52)

                if f["da3_depth"] and Path(f["da3_depth"]).exists():
                    da3 = np.load(f["da3_depth"]).astype(np.float32)
                    if da3.shape != (224, 224):
                        da3 = cv2.resize(da3, (224, 224), interpolation=cv2.INTER_LINEAR)
                else:
                    da3 = np.zeros((224, 224), dtype=np.float32)
                has_rs = False
                if f["rs_depth"] and Path(f["rs_depth"]).exists():
                    rs_full = np.load(f["rs_depth"]).astype(np.float32)
                    rs = cv2.resize(rs_full, (224, 224), interpolation=cv2.INTER_LINEAR)
                    has_rs = True
                else:
                    rs = np.zeros((224, 224), dtype=np.float32)

                if f["dinov3"]:
                    dino = np.load(f["dinov3"]).astype(np.float32)
                else:
                    dino = np.zeros(384, dtype=np.float32)

                seg_frames.append({"node": node, "world": world, "_da3": da3, "_rs": rs,
                                    "has_rs": has_rs, "dino": dino})
            self.segments.append({"subject": subj, "label": label, "frames": seg_frames})

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

        node = np.stack([f["node"] for f in frames], axis=0)  # (T, 33, 52)
        if self.augment:
            node = node + np.random.randn(*node.shape).astype(np.float32) * 0.005
            if np.random.rand() < 0.5:
                sx = np.random.uniform(0.92, 1.08)
                sy = np.random.uniform(0.92, 1.08)
                node[..., 0] *= sx; node[..., 1] *= sy
                node[..., 26] *= sx; node[..., 27] *= sy

        # Ergo features per window
        world_t = np.stack([f["world"] for f in frames], axis=0)
        ergo = compute_ergonomic_window(world_t)

        mid = self.window_t // 2
        da3 = frames[mid]["_da3"]; rs = frames[mid]["_rs"]; has_rs = frames[mid]["has_rs"]
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

        return {
            "node": torch.tensor(node, dtype=torch.float32),
            "depth": torch.tensor(depth_2ch, dtype=torch.float32),
            "dino": torch.tensor(dino, dtype=torch.float32),
            "ergo": torch.tensor(ergo, dtype=torch.float32),
            "label": seg["label"],
        }


def compute_depth_stats(ds, n=400):
    da3, rs = ds.collect_depth_for_stats(n_samples=n)
    return {
        "da3_mean": float(da3.mean()), "da3_std": float(da3.std() + 1e-8),
        "rs_mean":  float(rs.mean()),  "rs_std":  float(rs.std() + 1e-8),
    }


def evaluate(model, ds):
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    P, L = [], []
    with torch.no_grad():
        for b in ld:
            bd = {k: v.to(DEVICE) for k, v in b.items() if k != "label"}
            P.append(model(bd).argmax(1).cpu().numpy())
            lbl = b["label"]
            L.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl))
    P, L = np.concatenate(P), np.concatenate(L)
    cm = confusion_matrix(L, P, labels=list(range(5)))
    per_cls = {CLASSES[i]: float(cm[i, i] / cm[i].sum() * 100) if cm[i].sum() else 0.0 for i in range(5)}
    return {"acc": float((P == L).mean() * 100),
            "macro": float(np.mean(list(per_cls.values()))),
            "per_class": per_cls}


def train_fold(train_seg, val_seg, test_seg, bone_refs, fold_name):
    train_ds = WindowedDatasetErgo(train_seg, bone_refs, stride=TRAIN_STRIDE,
                                    augment=True, modality_dropout=0.1)
    val_ds = WindowedDatasetErgo(val_seg, bone_refs, stride=EVAL_STRIDE,
                                  augment=False, modality_dropout=0.0)
    test_ds = WindowedDatasetErgo(test_seg, bone_refs, stride=EVAL_STRIDE,
                                   augment=False, modality_dropout=0.0)

    stats = compute_depth_stats(train_ds)
    train_ds.set_depth_stats(stats); val_ds.set_depth_stats(stats); test_ds.set_depth_stats(stats)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = TemporalTeacherErgo(hidden=HIDDEN, n_classes=5, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW([
        {"params": model.skeleton.parameters(), "lr": 5e-4},
        {"params": model.depth.parameters(),    "lr": 1e-3},
        {"params": model.dino.parameters(),     "lr": 1e-3},
        {"params": model.ergo.parameters(),     "lr": 1e-3},
        {"params": model.fusion.parameters(),   "lr": 1e-3},
        {"params": model.head.parameters(),     "lr": 1e-3},
    ], weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = FocalLoss(gamma=FOCAL_GAMMA, weight=CLASS_WEIGHTS.to(DEVICE))

    best_val = -1
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        if ep <= WARMUP:
            for p in model.depth.parameters(): p.requires_grad_(False)
            for p in model.dino.parameters(): p.requires_grad_(False)
        elif ep == WARMUP + 1:
            for p in model.depth.parameters(): p.requires_grad_(True)
            for p in model.dino.parameters(): p.requires_grad_(True)

        for batch in train_ld:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(DEVICE) if isinstance(batch["label"], torch.Tensor) \
                else torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)
            logits = model(bd)
            loss = crit(logits, labels)
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
            print(f"  [{fold_name}] ep {ep:3d}/{EPOCHS}  val={val_res['macro']:.2f}%  best={best_val:.2f}%", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    test_res = evaluate(model, test_ds)
    return best_state, best_val, test_res, stats, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude_segments", default="")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    exclude = []
    for tok in args.exclude_segments.split(","):
        tok = tok.strip()
        if not tok: continue
        parts = tok.split("/")
        if len(parts) != 3:
            sys.exit(f"Bad exclude token '{tok}', expected subject/class/distance")
        exclude.append(tuple(parts))

    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    print("=" * 70)
    print(f"Day 7 — Day 2 architecture + ergonomic features (from scratch)  suffix='{args.out_suffix}'")
    if exclude:
        print(f"  excluding: {exclude}")
    print("=" * 70)

    segments, rows_per_subj = load_segments(exclude_keys=exclude)
    bone_refs = {s: compute_subject_bone_reference(rows_per_subj[s]) for s in SUBJECTS}
    print(f"Loaded {len(segments)} segments, computed bone refs for {len(bone_refs)} subjects")

    fold_results = []
    t0 = time.time()
    for fi, held_out in enumerate(SUBJECTS):
        train_pool = [s for s in SUBJECTS if s != held_out]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]

        print(f"\n{'=' * 70}")
        print(f"FOLD {fi+1}/5  held_out={held_out}  val={val_subj}")
        print(f"  elapsed: {(time.time() - t0)/60:.1f} min")
        print('=' * 70)

        train_seg = {k: v for k, v in segments.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in segments.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in segments.items() if k[0] == held_out}

        best_state, best_val, test_res, stats, n_params = train_fold(
            train_seg, val_seg, test_seg, bone_refs, f"fold{fi+1}_{held_out}"
        )
        torch.save({"model_state": best_state, "test": test_res, "val_macro": best_val,
                    "depth_stats": stats, "held_out": held_out, "n_params": n_params,
                    "bone_refs": bone_refs}, CKPT_DIR / f"{held_out}{args.out_suffix}_best.pt")
        print(f"  [{held_out}] val={best_val:.2f}% test_macro={test_res['macro']:.2f}% per_class={test_res['per_class']}", flush=True)

        fold_results.append({
            "fold": fi+1, "held_out": held_out, "val_subject": val_subj,
            "val_macro": best_val, "test": test_res, "n_params": n_params,
        })
        with open(OUT_DIR / f"day7_results{args.out_suffix}.json", "w") as f:
            json.dump({"folds": fold_results}, f, indent=2)

    macros = [f["test"]["macro"] for f in fold_results]
    print(f"\n{'=' * 70}")
    print(f"DAY 7 DONE — {(time.time() - t0)/60:.1f} min")
    print(f"  Mean macro: {np.mean(macros):.2f} ± {np.std(macros):.2f}")
    print('=' * 70)

    md = ["# Day 7 — Day 2 Architecture + Ergonomic Features (from scratch)\n",
          "Hyperparameters identical to Day 2 (T=7, hybrid 52-feat skel, focal γ=2 fh-weight=3,",
          "body-shape aug, 120 epochs). Adds ergonomic features as a 4th branch.\n",
          f"Suffix: '{args.out_suffix}' (excluded: {exclude})\n",
          "## Per-fold\n",
          "| held_out | val | val_macro | test_macro | fh | left | right | sit | sl |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in fold_results:
        pc = f["test"]["per_class"]
        md.append(f"| {f['held_out']} | {f['val_subject']} | {f['val_macro']:.1f}% | "
                  f"**{f['test']['macro']:.1f}%** | "
                  f"{pc['forward_head']:.0f} | {pc['left_leaning']:.0f} | "
                  f"{pc['right_leaning']:.0f} | {pc['sit_straight']:.0f} | "
                  f"{pc['slouched_posture']:.0f} |")
    md.append(f"| **mean ± std** | | | **{np.mean(macros):.2f} ± {np.std(macros):.2f}** | | | | | |")
    md.append("")
    md.append("## Per-class mean across folds\n")
    md.append("| class | mean | std | min | max |")
    md.append("|---|---|---|---|---|")
    for c in CLASSES:
        vs = [f["test"]["per_class"][c] for f in fold_results]
        md.append(f"| {c} | {np.mean(vs):.1f} | {np.std(vs):.1f} | {min(vs):.0f} | {max(vs):.0f} |")
    md.append("")
    md.append("## Comparison\n")
    md.append("| run | mean macro | std |")
    md.append("|---|---|---|")
    md.append("| Day 0 (single-frame) | 74.12 | 9.70 |")
    md.append("| Day 1 (T=15 bone) | 72.00 | 3.01 |")
    md.append("| Day 2 (T=7 hybrid focal) | 71.98 | 6.47 |")
    md.append(f"| **Day 7 (Day 2 + ergo){args.out_suffix}** | **{np.mean(macros):.2f}** | **{np.std(macros):.2f}** |")
    md.append("")

    out_md = OUT_DIR / f"day7_results{args.out_suffix}.md"
    out_md.write_text("\n".join(md))
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
