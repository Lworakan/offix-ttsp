"""
Day-2 SOTA refactor (targeted fixes for the Day-1 fh regression).

Changes vs Day 1:
  1. Shorter window: T=7 (was 15) — preserves brief fh moments
  2. Hybrid 52-feat skeleton: raw shoulder-norm (26) + bone-normalized (26) concat
  3. Focal loss, gamma=2.0, class weight forward_head=3.0
  4. Body-shape aug disabled on z-axis (x, y only) — preserves head depth signal
  5. 120 epochs, train stride 2 (was 80 ep, stride 5) — ~3x more samples / epoch

Run via:
  tmux new -s sota2
  python3 scripts/ms_62_day2_targeted.py 2>&1 | tee /tmp/day2.log
  Ctrl+B  D
"""

import csv, sys, time, json, warnings
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

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "multisubject" / "webcam_dataset.csv"
DINOV3_ROOT = ROOT / "data" / "multisubject" / "dinov3_features"
OUT_DIR = ROOT / "outputs"
CKPT_DIR = ROOT / "checkpoints" / "day2_targeted"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DROP_SUBJECTS = {"pan"}
SUBJECTS = ["fu", "mukrop", "nonny", "boom", "peemai"]
CLASSES = ["forward_head", "left_leaning", "right_leaning", "sit_straight", "slouched_posture"]
EXCLUDE_CLASSES = {"freestyle_sitting", "rounded_shoulders"}
FH_IDX = 0  # forward_head

# ── Day 2 hyperparameters ─────────────────────────────────────────────────
WINDOW_T = 7
TRAIN_STRIDE = 2
EVAL_STRIDE = 7
EPOCHS = 120
WARMUP = 15
BATCH = 32
HIDDEN = 128
DROPOUT = 0.15
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.5], dtype=torch.float32)  # fh=3, sl=1.5

# Body topology
BODY_EDGES = [
    (0, 2), (0, 5), (2, 7), (5, 8),
    (11, 12), (0, 11), (0, 12), (7, 11), (8, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (11, 24), (12, 23),
]
ADJ_LIST = {i: [] for i in range(33)}
for a, b in BODY_EDGES:
    ADJ_LIST[a].append(b); ADJ_LIST[b].append(a)

REF_BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (0, 11), (0, 12), (7, 8),
]


def compute_subject_bone_reference(rows_for_subject):
    lengths = defaultdict(list)
    sit_rows = [r for r in rows_for_subject if r["class"] == "sit_straight"]
    if not sit_rows: sit_rows = rows_for_subject
    for r in sit_rows[:200]:
        lm = r.get("webcam_landmarks", "")
        if not lm or not Path(lm).exists(): continue
        world = np.load(lm).astype(np.float32)
        for i, (a, b) in enumerate(REF_BONES):
            lengths[i].append(float(np.linalg.norm(world[a] - world[b])))
    ref = {i: float(np.median(lengths[i])) if lengths[i] else 1.0 for i in range(len(REF_BONES))}
    ref["scale"] = ref[0]
    return ref


# ── Hybrid feature builder: 52 = raw 26 + bone-normalized 26 ──────────────

def _features_one_norm(world, img, scale_world):
    """26-feature output normalized by `scale_world`."""
    sw_world = max(scale_world, 1e-6)
    sh_mid_w = (world[11] + world[12]) / 2
    wn = (world - sh_mid_w) / sw_world
    sh_mid_i = (img[11] + img[12]) / 2
    si = max(np.linalg.norm(img[11] - img[12]), 1e-6)
    imn = (img - sh_mid_i) / si

    feats = []
    for j in range(33):
        feat = list(wn[j]) + list(imn[j])
        neighbors = ADJ_LIST.get(j, [])
        bone_lens = [float(np.linalg.norm(wn[j] - wn[nb])) for nb in neighbors[:6]]
        while len(bone_lens) < 6: bone_lens.append(0.0)
        feat.extend(bone_lens)
        for nb in neighbors[:4]:
            diff = wn[nb] - wn[j]
            n = np.linalg.norm(diff) + 1e-6
            feat.extend([float(diff[0] / n), float(diff[1] / n), float(diff[2] / n)])
        while len(feat) < 23: feat.append(0.0)
        norm_j = float(np.linalg.norm(wn[j]))
        feat.append(-float(wn[j, 1]) / (norm_j + 1e-6) if norm_j > 1e-6 else 0.0)
        feat.append(norm_j)
        feat.append(0.0)
        feats.append(feat[:26])
    return np.array(feats, dtype=np.float32)


def compute_node_features_hybrid(world, img, bone_ref):
    """52 features per joint = raw shoulder-width norm (26) + bone-normalized (26)."""
    raw = _features_one_norm(world, img, np.linalg.norm(world[11] - world[12]))
    bone = _features_one_norm(world, img, bone_ref["scale"])
    return np.concatenate([raw, bone], axis=1)  # (33, 52)


# ── Dataset ───────────────────────────────────────────────────────────────

def load_segments():
    segments = defaultdict(list)
    rows_per_subj = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["subject"] in DROP_SUBJECTS: continue
            if r["class"] in EXCLUDE_CLASSES: continue
            if r["class"] not in CLASSES: continue
            lm = r.get("webcam_landmarks", "")
            if not lm or not Path(lm).exists(): continue
            stem = r.get("frame_stem", "0")
            dino_path = DINOV3_ROOT / r["subject"] / r["class"] / r.get("distance", "nom") / f"{stem}_dino.npy"
            key = (r["subject"], r["class"], r.get("distance", "nom"))
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


class WindowedDataset(Dataset):
    def __init__(self, segments, bone_refs, stats=None,
                 window_t=WINDOW_T, stride=TRAIN_STRIDE,
                 augment=False, modality_dropout=0.0):
        self.bone_refs = bone_refs
        self.window_t = window_t
        self.augment = augment
        self.modality_dropout = modality_dropout
        self._depth_stats = stats
        self.segments = []

        n_total = 0
        for key, frames in segments.items():
            subj = key[0]
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

                seg_frames.append({"node": node, "_da3": da3, "_rs": rs,
                                   "has_rs": has_rs, "dino": dino})
                n_total += 1
            self.segments.append({"subject": subj, "label": label, "frames": seg_frames})

        self.windows = []
        for si, seg in enumerate(self.segments):
            n = len(seg["frames"])
            if n < window_t: continue
            for start in range(0, n - window_t + 1, stride):
                self.windows.append((si, start))
        print(f"  loaded {n_total} frames, {len(self.windows)} windows (T={window_t}, stride={stride})")

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
            # Body-shape aug: x, y only (NO z) — preserves head depth signal
            if np.random.rand() < 0.5:
                sx = np.random.uniform(0.92, 1.08)
                sy = np.random.uniform(0.92, 1.08)
                # Scale world coords (first 3 of each 26-block, both halves)
                node[..., 0] *= sx; node[..., 1] *= sy
                node[..., 26] *= sx; node[..., 27] *= sy

        mid = self.window_t // 2
        da3 = frames[mid]["_da3"]; rs = frames[mid]["_rs"]; has_rs = frames[mid]["has_rs"]
        s = self._depth_stats
        if s is not None:
            da3_n = (da3 - s['da3_mean']) / s['da3_std']
            rs_n = (rs - s['rs_mean']) / s['rs_std']
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
            "label": seg["label"],
        }


# ── Model ─────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, adj):
        super().__init__()
        self.adj = adj
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        adj = self.adj.to(x.device)
        h = torch.einsum('nm,bmd->bnd', adj, x)
        h = self.linear(h)
        B, N, D = h.shape
        h = self.bn(h.reshape(B * N, D)).reshape(B, N, D)
        return F.gelu(h) + self.skip(x)


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
        self.t_attn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.out_dim = hidden

    def forward(self, x):
        B, T, J, F_ = x.shape
        x = x.reshape(B * T, J, F_)
        for layer in self.gcn:
            x = layer(x); x = self.drop(x)
        x = x.mean(dim=1).reshape(B, T, -1)
        attn = torch.softmax(self.t_attn(x), dim=1)
        return (x * attn).sum(dim=1)


class DepthBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
    def forward(self, x): return F.gelu(self.bn(self.conv(x)))


class DepthBranch(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.b1 = DepthBlock(2, 32, 2); self.b2 = DepthBlock(32, 64, 2)
        self.b3 = DepthBlock(64, 128, 2); self.b4 = DepthBlock(128, 192, 2)
        self.b5 = DepthBlock(192, hidden, 2); self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x); x = self.b5(x)
        return self.gap(x).flatten(1)


class DINOBranch(nn.Module):
    def __init__(self, in_dim=384, hidden=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, hidden), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, n_modalities=3, hidden=128):
        super().__init__()
        self.gates = nn.Linear(n_modalities * hidden, n_modalities)
    def forward(self, mods):
        x = torch.cat(mods, dim=1)
        gw = torch.softmax(self.gates(x), dim=1)
        fused = sum(gw[:, i:i+1] * mods[i] for i in range(len(mods)))
        return torch.cat([fused, x], dim=1)


class TemporalTeacher(nn.Module):
    def __init__(self, hidden=128, n_classes=5, dropout=0.15):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=hidden, dropout=dropout)
        self.depth = DepthBranch(hidden=hidden)
        self.dino = DINOBranch(in_dim=384, hidden=hidden, dropout=dropout)
        self.fusion = GatedFusion(3, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch):
        sk = self.skeleton(batch["node"])
        dp = self.depth(batch["depth"])
        di = self.dino(batch["dino"])
        return self.head(self.fusion([sk, dp, di]))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight,
                             label_smoothing=self.ls, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


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
            L.append(b["label"].numpy() if isinstance(b["label"], torch.Tensor) else np.array(b["label"]))
    P, L = np.concatenate(P), np.concatenate(L)
    cm = confusion_matrix(L, P, labels=list(range(5)))
    per_cls = {CLASSES[i]: float(cm[i, i] / cm[i].sum() * 100) if cm[i].sum() else 0.0 for i in range(5)}
    return {"acc": float((P == L).mean() * 100),
            "macro": float(np.mean(list(per_cls.values()))),
            "per_class": per_cls, "cm": cm.tolist()}


def train_fold(train_seg, val_seg, test_seg, bone_refs, fold_name):
    print(f"\n[{fold_name}] building datasets…")
    train_ds = WindowedDataset(train_seg, bone_refs, stride=TRAIN_STRIDE,
                                augment=True, modality_dropout=0.1)
    val_ds = WindowedDataset(val_seg, bone_refs, stride=EVAL_STRIDE,
                              augment=False, modality_dropout=0.0)
    test_ds = WindowedDataset(test_seg, bone_refs, stride=EVAL_STRIDE,
                               augment=False, modality_dropout=0.0)

    stats = compute_depth_stats(train_ds)
    train_ds.set_depth_stats(stats); val_ds.set_depth_stats(stats); test_ds.set_depth_stats(stats)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = TemporalTeacher(hidden=HIDDEN, n_classes=5, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params")

    opt = torch.optim.AdamW([
        {"params": model.skeleton.parameters(), "lr": 5e-4},
        {"params": model.depth.parameters(),    "lr": 1e-3},
        {"params": model.dino.parameters(),     "lr": 1e-3},
        {"params": model.fusion.parameters(),   "lr": 1e-3},
        {"params": model.head.parameters(),     "lr": 1e-3},
    ], weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = FocalLoss(gamma=FOCAL_GAMMA, weight=CLASS_WEIGHTS.to(DEVICE))

    best_val = -1
    best_state = None
    best_fh = -1

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
            best_fh = val_res["per_class"]["forward_head"]
        if ep % 5 == 0 or ep == 1:
            print(f"  [{fold_name}] ep {ep:3d}/{EPOCHS}  val={val_res['macro']:.2f}%  "
                  f"fh={val_res['per_class']['forward_head']:.0f}  best_val={best_val:.2f}%", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    test_res = evaluate(model, test_ds)
    return best_state, best_val, test_res, stats


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    print("=" * 70)
    print("Day 2 SOTA refactor — targeted fixes")
    print(f"  T={WINDOW_T} train_stride={TRAIN_STRIDE} eval_stride={EVAL_STRIDE}")
    print(f"  hybrid 52-feat skeleton | focal loss g={FOCAL_GAMMA} | fh weight={CLASS_WEIGHTS[0]}")
    print(f"  body aug: x,y only (no z) | epochs={EPOCHS}")
    print("=" * 70)

    print("\nLoading segments…")
    segments, rows_per_subj = load_segments()
    print(f"  {len(segments)} segments")

    print("\nComputing per-subject bone references…")
    bone_refs = {}
    for s in SUBJECTS:
        bone_refs[s] = compute_subject_bone_reference(rows_per_subj[s])
        print(f"  {s}: shoulder={bone_refs[s]['scale']:.4f}m")

    fold_results = []
    t0 = time.time()
    for fi, held_out in enumerate(SUBJECTS):
        train_pool = [s for s in SUBJECTS if s != held_out]
        val_subj = train_pool[fi % len(train_pool)]
        train_subjects = [s for s in train_pool if s != val_subj]

        print(f"\n{'=' * 70}")
        print(f"FOLD {fi+1}/5  held_out={held_out}  val={val_subj}  train={train_subjects}")
        print(f"  elapsed: {(time.time() - t0)/60:.1f} min")
        print('=' * 70)

        train_seg = {k: v for k, v in segments.items() if k[0] in train_subjects}
        val_seg = {k: v for k, v in segments.items() if k[0] == val_subj}
        test_seg = {k: v for k, v in segments.items() if k[0] == held_out}

        best_state, best_val, test_res, stats = train_fold(
            train_seg, val_seg, test_seg, bone_refs, f"fold{fi+1}_{held_out}"
        )
        ckpt = CKPT_DIR / f"{held_out}_best.pt"
        torch.save({"model_state": best_state, "val_macro": best_val,
                    "test": test_res, "depth_stats": stats,
                    "held_out": held_out, "val_subject": val_subj}, ckpt)
        print(f"  [{held_out}] val={best_val:.2f}%  test_macro={test_res['macro']:.2f}%  "
              f"per_class={test_res['per_class']}", flush=True)

        fold_results.append({
            "fold": fi+1, "held_out": held_out, "val_subject": val_subj,
            "val_macro": best_val, "test": test_res,
        })
        with open(OUT_DIR / "day2_loso_results.json", "w") as f:
            json.dump({"folds": fold_results}, f, indent=2)

    print(f"\n{'=' * 70}\nDay 2 LOSO COMPLETE — {(time.time() - t0)/60:.1f} min\n{'=' * 70}")
    macros = [f["test"]["macro"] for f in fold_results]
    md = ["# Day 2 LOSO — Targeted Fixes\n",
          "Changes vs Day 1:\n",
          "- T=7 (was 15)  | hybrid 52-feat (raw + bone-norm)  | focal γ=2 fh-weight=3",
          "- body-shape aug x,y only  | 120 epochs stride 2\n",
          "## Per-fold\n",
          "| held_out | val | val_macro | test_acc | test_macro | fh | left | right | sit | sl |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fold_results:
        pc = f["test"]["per_class"]
        md.append(f"| {f['held_out']} | {f['val_subject']} | {f['val_macro']:.1f}% | "
                  f"{f['test']['acc']:.1f}% | **{f['test']['macro']:.1f}%** | "
                  f"{pc['forward_head']:.0f} | {pc['left_leaning']:.0f} | "
                  f"{pc['right_leaning']:.0f} | {pc['sit_straight']:.0f} | "
                  f"{pc['slouched_posture']:.0f} |")
    md.append(f"| **mean ± std** | | | | **{np.mean(macros):.2f} ± {np.std(macros):.2f}** | | | | | |")
    md.append("\n## Per-class mean across folds\n")
    md.append("| class | mean | std | min | max |")
    md.append("|---|---|---|---|---|")
    for c in CLASSES:
        vs = [f["test"]["per_class"][c] for f in fold_results]
        md.append(f"| {c} | {np.mean(vs):.1f} | {np.std(vs):.1f} | {min(vs):.0f} | {max(vs):.0f} |")
    md.append(f"\n## Trend\n")
    md.append(f"- Day 0 (single-frame, 5 subj): 74.12 ± 9.70%")
    md.append(f"- Day 1 (T=15, bone, focal off): 72.00 ± 3.01%")
    md.append(f"- **Day 2 (T=7, hybrid, focal):  {np.mean(macros):.2f} ± {np.std(macros):.2f}%**")
    md.append(f"- Δ vs Day 1: **{np.mean(macros) - 72.00:+.2f} pp**")
    md.append(f"- Δ vs Day 0: **{np.mean(macros) - 74.12:+.2f} pp**")

    (OUT_DIR / "day2_loso_results.md").write_text("\n".join(md))
    print(f"Report: {OUT_DIR / 'day2_loso_results.md'}")
    print(f"Mean ± std: {np.mean(macros):.2f} ± {np.std(macros):.2f}")


if __name__ == "__main__":
    main()
