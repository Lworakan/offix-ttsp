"""
ms_173 — Final 9-Subject LOSO: Skel + L/R-Anchored Clinical Depth + Compensatory.

Three feature streams, all L/R-anchored:
  1. Skeleton GCN (52-dim temporal, handles L/R perfectly)
  2. Compensatory: sh/ear ratio (shoulder rounding from Upper Crossed Syndrome)
  3. Clinical depth: HFD + CVA from DepthPro (direct fh/sit measurement, sagittal only)

All features L/R-anchored to remove subject-specific offset.
NOM-only train, NOM-only eval, 9-fold LOSO.

Experiments:
  E1a: skel only (baseline)
  E1b: skel + anchored sh/ear (compensatory only)
  E1c: skel + anchored HFD+CVA (depth only)
  E1d: skel + anchored sh/ear + HFD + CVA (full)
  E1e: skel + raw (unanchored) sh/ear + HFD + CVA (no anchor ablation)

Usage:
  python3 scripts/ms_173_final_9subj.py --seed 42
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
from ms_160_nom_candidates import compute_clinical_3d, CLINICAL_3D_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "outputs"

SUBJECTS_9 = ["fu", "mukrop", "nonny", "boom", "peemai", "pan", "namoon", "mai", "money"]
OLD_SUBJECTS = ["fu", "mukrop", "nonny", "boom", "peemai", "pan"]
NEW_SUBJECTS = ["namoon", "mai", "money"]

OLD_LM = ROOT / "data" / "multisubject" / "webcam_landmarks"
NEW_LM = ROOT / "data" / "multisubject" / "webcam_landmarks_new"
OLD_DP = ROOT / "data" / "multisubject" / "depthpro"
NEW_DP = ROOT / "data" / "multisubject" / "depthpro_new"

TRAIN_STRIDE = 3
EVAL_STRIDE = 1
EPOCHS = 120
BATCH = 16


def load_nom_segments():
    """Load NOM-only segments for all 9 subjects."""
    segments = defaultdict(list)
    rows_per_subj = defaultdict(list)
    for subj in SUBJECTS_9:
        lm_root = OLD_LM if subj in OLD_SUBJECTS else NEW_LM
        dp_root = OLD_DP if subj in OLD_SUBJECTS else NEW_DP
        for cls in CLASSES_4:
            lm_dir = lm_root / subj / cls / 'nom'
            if not lm_dir.exists(): continue
            for lm_file in sorted(lm_dir.glob("*_landmarks.npy")):
                stem = lm_file.stem.replace("_landmarks", "")
                img_file = lm_dir / f"{stem}_landmarks_img.npy"
                if not img_file.exists(): continue
                w = np.load(lm_file)
                if np.all(w == 0): continue
                dp_file = dp_root / subj / cls / 'nom' / f"{stem}_depth.npy"
                segments[(subj, cls, 'nom')].append({
                    "lm_world": str(lm_file), "lm_img": str(img_file),
                    "depthpro": str(dp_file) if dp_file.exists() else None,
                    "subject": subj, "frame": stem,
                })
                rows_per_subj[subj].append({"subject": subj, "class": cls,
                                             "webcam_landmarks": str(lm_file)})
    for k in segments:
        segments[k].sort(key=lambda f: f["frame"])
    return segments, rows_per_subj


def compute_features(world, img, dp_map):
    """Compute compensatory (1-dim) + clinical depth (2-dim) features."""
    # Compensatory: sh/ear ratio
    sw = np.linalg.norm(img[11] - img[12])
    ew = np.linalg.norm(img[7] - img[8])
    sh_ear = sw / max(ew, 1e-6)
    # Neck length (image-space)
    sh_mid = (img[11] + img[12]) / 2
    neck_len = np.linalg.norm(img[0] - sh_mid)
    # Clinical depth: HFD + CVA from DepthPro (indices 0 and 1 of clinical_3d)
    if dp_map is not None:
        c3d = compute_clinical_3d(img, dp_map)
        hfd = c3d[0]  # Head Forward Distance
        cva = c3d[1]  # Craniovertebral Angle
    else:
        hfd = 0.0; cva = 0.0
    return np.array([sh_ear, neck_len, hfd, cva], dtype=np.float32)


def compute_lr_anchors(segments):
    """Per-subject feature anchors from L+R NOM segments."""
    anchors = {}
    for subj in SUBJECTS_9:
        lr_feats = []
        for cls in ['left_leaning', 'right_leaning']:
            key = (subj, cls, 'nom')
            if key not in segments: continue
            for f in segments[key][:25]:
                w = np.load(f["lm_world"]).astype(np.float32)
                img = np.load(f["lm_img"]).astype(np.float32)
                dp = None
                if f["depthpro"] and Path(f["depthpro"]).exists():
                    dp = np.load(f["depthpro"]).astype(np.float32)
                    if dp.shape != (224, 224): dp = cv2.resize(dp, (224, 224))
                lr_feats.append(compute_features(w, img, dp))
        if lr_feats:
            anchors[subj] = np.mean(lr_feats, axis=0).astype(np.float32)
    return anchors


class Dataset9Subj(torch.utils.data.Dataset):
    def __init__(self, segment_dict, bone_refs, anchors, stride=TRAIN_STRIDE,
                 augment=False, use_anchor=True, feature_mode='full'):
        """
        feature_mode: 'skel_only', 'comp', 'depth', 'full', 'full_raw'
          skel_only: no extra features
          comp: sh/ear ratio only (index 0)
          depth: HFD + CVA only (indices 2,3)
          full: sh/ear + HFD + CVA (indices 0,2,3)
          full_raw: same as full but no anchoring
        """
        self.window_t = WINDOW_T
        self.augment = augment
        self.feature_mode = feature_mode
        self.segments = []
        for key, frames in segment_dict.items():
            subj = key[0]
            if subj not in bone_refs: continue
            label = CLASSES_4.index(key[1])
            ref = bone_refs[subj]
            anchor = anchors.get(subj, np.zeros(4, dtype=np.float32))
            seg_frames = []
            for f in frames:
                w = np.load(f["lm_world"]).astype(np.float32)
                img = np.load(f["lm_img"]).astype(np.float32)
                dp = None
                if f["depthpro"] and Path(f["depthpro"]).exists():
                    dp = np.load(f["depthpro"]).astype(np.float32)
                    if dp.shape != (224, 224): dp = cv2.resize(dp, (224, 224))
                node = compute_node_features_hybrid(w, img, ref)
                raw_feats = compute_features(w, img, dp)
                if use_anchor:
                    anchored = raw_feats - anchor
                else:
                    anchored = raw_feats
                seg_frames.append({"node": node, "feats": anchored})
            self.segments.append({"subject": subj, "label": label, "frames": seg_frames})
        self.windows = []
        for si, seg in enumerate(self.segments):
            n = len(seg["frames"])
            if n < WINDOW_T: continue
            for start in range(0, n - WINDOW_T + 1, stride):
                self.windows.append((si, start))

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        si, start = self.windows[idx]
        seg = self.segments[si]
        frames = seg["frames"][start:start+self.window_t]
        node = np.stack([f["node"] for f in frames])
        if self.augment:
            node = node + np.random.randn(*node.shape).astype(np.float32) * 0.005
        mid = self.window_t // 2
        feats = frames[mid]["feats"]
        # Select features based on mode
        # feats = [sh_ear, neck_len, hfd, cva]
        # Verified: HFD delta (fh-sit) negative for ALL 9 subjects ✓
        # Verified: CVA delta (fh-sit) positive for ALL 9 subjects ✓
        # Verified: sh/ear delta (fh-sit) negative for ALL 9 subjects ✓ (but absolute values overlap)
        if self.feature_mode == 'comp':
            extra = feats[0:1]       # sh/ear only (1-dim)
        elif self.feature_mode == 'depth':
            extra = feats[2:4]       # HFD + CVA (2-dim, both verified consistent)
        elif self.feature_mode in ('full', 'full_raw'):
            extra = feats[[0, 2, 3]] # sh/ear + HFD + CVA (3-dim)
        else:
            extra = np.array([], dtype=np.float32)
        return {
            "node": torch.tensor(node, dtype=torch.float32),
            "extra": torch.tensor(extra, dtype=torch.float32),
            "label": seg["label"],
        }


class SkelExtra(nn.Module):
    def __init__(self, extra_dim, h=HIDDEN, nc=4, do=0.15):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(
            nn.Linear(h + extra_dim, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk = self.skeleton(batch["node"])
        if batch["extra"].shape[1] > 0:
            return self.cls_head(torch.cat([sk, batch["extra"]], dim=1))
        return self.cls_head(sk)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def evaluate_model(model, ds, save_preds_path=None):
    from sklearn.metrics import confusion_matrix
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds, labels, logits_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            lg = model(bd).cpu().numpy()
            logits_all.append(lg)
            preds.extend(lg.argmax(-1))
            labels.extend(batch["label"].numpy())
    preds = np.array(preds); labels = np.array(labels)
    if save_preds_path is not None:
        np.savez(save_preds_path, labels=labels, preds=preds,
                 logits=np.concatenate(logits_all))
    cm = confusion_matrix(labels, preds, labels=list(range(4)))
    pc = {c: float(cm[i, i] / max(cm[i].sum(), 1) * 100) for i, c in enumerate(CLASSES_4)}
    return {"macro": float(np.mean(list(pc.values()))), "per_class": pc}


def train_fold(extra_dim, train_ds, val_ds, test_ds, seed, save_preds_path=None):
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)
    model = SkelExtra(extra_dim).to(DEVICE)
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
    return evaluate_model(model, test_ds, save_preds_path=save_preds_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated experiments to run (e.g. E1a,E1d,E1e)")
    ap.add_argument("--save_preds", action="store_true",
                    help="dump per-sample predictions per (exp,fold) to outputs/")
    args = ap.parse_args()
    set_seed(args.seed)

    print("Loading 9 subjects (NOM only)...", flush=True)
    all_segs, rows = load_nom_segments()
    brefs = {s: compute_subject_bone_reference(rows[s]) for s in SUBJECTS_9 if s in rows}
    anchors = compute_lr_anchors(all_segs)

    for s in SUBJECTS_9:
        n = sum(len(v) for k, v in all_segs.items() if k[0] == s)
        a = anchors.get(s, np.zeros(4))
        print(f"  {s}: {n} frames  anchor: sh_ear={a[0]:.3f} HFD={a[2]:+.4f} CVA={a[3]:.1f}", flush=True)

    EXPERIMENTS = {
        'E1a': {'mode': 'skel_only', 'anchor': True,  'extra_dim': 0, 'desc': 'skel only (baseline)'},
        'E1b': {'mode': 'comp',      'anchor': True,  'extra_dim': 1, 'desc': 'skel + anchored sh/ear'},
        'E1c': {'mode': 'depth',     'anchor': True,  'extra_dim': 2, 'desc': 'skel + anchored HFD+CVA'},
        'E1d': {'mode': 'full',      'anchor': True,  'extra_dim': 3, 'desc': 'skel + anchored sh/ear+HFD+CVA (FULL)'},
        'E1e': {'mode': 'full_raw',  'anchor': False, 'extra_dim': 3, 'desc': 'skel + raw sh/ear+HFD+CVA (no anchor)'},
    }

    all_results = {}
    t0 = time.time()

    only_set = set(args.only.split(",")) if args.only else None

    for ename, ecfg in EXPERIMENTS.items():
        if only_set and ename not in only_set:
            continue
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

            use_anch = ecfg['anchor']
            fmode = ecfg['mode'] if ecfg['mode'] != 'full_raw' else 'full'
            train_ds = Dataset9Subj(train_seg, brefs, anchors, stride=TRAIN_STRIDE,
                                    augment=True, use_anchor=use_anch, feature_mode=fmode)
            val_ds = Dataset9Subj(val_seg, brefs, anchors, stride=EVAL_STRIDE,
                                  augment=False, use_anchor=use_anch, feature_mode=fmode)
            test_ds = Dataset9Subj(test_seg, brefs, anchors, stride=EVAL_STRIDE,
                                   augment=False, use_anchor=use_anch, feature_mode=fmode)

            spath = (OUT_DIR / f"ms173_predictions_{ename}_{held_out}.npz") if args.save_preds else None
            test_res = train_fold(ecfg['extra_dim'], train_ds, val_ds, test_ds,
                                   args.seed + fi, save_preds_path=spath)
            pc = test_res["per_class"]
            fh = pc["forward_head"]; sit = pc["sit_straight"]
            both = "BOTH>=90" if fh >= 90 and sit >= 90 else ""
            all4 = "ALL4>=90" if all(v >= 90 for v in pc.values()) else ""
            tag = "(NEW)" if held_out in NEW_SUBJECTS else ""
            print(f"  [{held_out}] {test_res['macro']:.1f}%  fh={fh:.0f} sit={sit:.0f} "
                  f"L={pc['left_leaning']:.0f} R={pc['right_leaning']:.0f}  {both} {all4} {tag}", flush=True)
            fold_results.append({"held_out": held_out, "test": test_res})

        # Summary
        old_f = [f for f in fold_results if f["held_out"] in OLD_SUBJECTS and f["held_out"] != "pan"]
        new_f = [f for f in fold_results if f["held_out"] in NEW_SUBJECTS]
        all_f = [f for f in fold_results if f["held_out"] != "pan"]
        m_all = [f["test"]["macro"] for f in all_f]
        both_all = sum(1 for f in all_f if f["test"]["per_class"]["forward_head"] >= 90
                       and f["test"]["per_class"]["sit_straight"] >= 90)
        all4_all = sum(1 for f in all_f if all(f["test"]["per_class"][c] >= 90 for c in CLASSES_4))

        print(f"\n  [{ename}] 8-subj(excl pan): {np.mean(m_all):.1f}%  "
              f"both_fh_sit>=90: {both_all}/8  all4>=90: {all4_all}/8", flush=True)
        all_results[ename] = {"folds": fold_results, "mean_8": float(np.mean(m_all)),
                               "both_90": both_all, "all4_90": all4_all, "desc": ecfg['desc']}

    # Final
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL (9-subj LOSO, NOM-only, {elapsed:.0f} min)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Exp':>5} {'Desc':>45} {'8-sub':>6} {'both':>5} {'all4':>5}", flush=True)
    for e, r in all_results.items():
        print(f"  {e}  {r['desc']:>45} {r['mean_8']:>5.1f}  {r['both_90']:>3}/8  {r['all4_90']:>3}/8", flush=True)

    with open(OUT_DIR / "ms173_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'ms173_final.json'}", flush=True)


if __name__ == "__main__":
    main()
