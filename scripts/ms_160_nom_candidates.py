"""
ms_160 — Train ALL candidate architectures (A-K) on NOM-only data.

NOM = standard laptop distance = deployment-realistic.
Close/far are recording protocol artifacts removed from both train and eval.

11 candidates evaluated sequentially with 6-fold LOSO.
Each trains on ~80 windows (NOM-only, 4 subjects per fold).

Usage:
  python3 scripts/ms_160_nom_candidates.py --seed 42
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

from ms_140_common import (
    SUBJECTS_6, CLASSES_4, ERGO_DIM, DEPTH_FEAT_DIM, WINDOW_T,
    load_segments_4c_6s, load_side_supervision,
    WindowedDatasetV2, evaluate, FocalLoss,
)
from ms_62_day2_targeted import compute_subject_bone_reference, compute_node_features_hybrid
from ms_73_day2plus_ergo import HIDDEN, DROPOUT, TemporalSkeletonBranch
from ms_128_ecgf_ldc import (
    LandmarkDepthBranch, ErgonomicBranch30, LAMBDA_RULA, LAMBDA_FH,
    extract_landmark_depth_40, compute_body_frame_30,
)
from ms_84_ecgf import ErgonomicGatedFusion
from ms_114_ecgf_xs import set_seed
from ms_140_train import compute_subject_anchors, AnchoredDataset, ANCHOR_FEAT_IDX
from ms_156_train import extract_landmark_depth_40_normalized

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "outputs"
DEPTHPRO_ROOT = ROOT / "data" / "multisubject" / "depthpro"

TRAIN_STRIDE = 3  # denser for NOM-only (fewer segments)
EVAL_STRIDE = 1   # stride=1 for reliable measurement
EPOCHS = 120
BATCH = 16  # smaller batch for smaller dataset
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
FOCAL_GAMMA = 2.0
CLINICAL_3D_DIM = 5


# ═══════════════════════════════════════════════════════════════════════
# Clinical 3D features (from ms_155)
# ═══════════════════════════════════════════════════════════════════════
def compute_clinical_3d(img_lm, depthpro_map):
    feats = np.zeros(CLINICAL_3D_DIM, dtype=np.float32)
    H, W = depthpro_map.shape
    def get_3d(idx):
        u = float(np.clip(img_lm[idx, 0], 0.001, 0.999))
        v = float(np.clip(img_lm[idx, 1], 0.001, 0.999))
        pu = max(2, min(W-3, int(round(u*W)))); pv = max(2, min(H-3, int(round(v*H))))
        z_inv = float(np.median(depthpro_map[pv-2:pv+3, pu-2:pu+3]))
        if z_inv <= 0.01: z_inv = 0.5
        d = 1.0 / z_inv
        return np.array([(u-0.5)*d, (v-0.5)*d, d])
    try:
        l_ear=get_3d(7); r_ear=get_3d(8); l_eye=get_3d(2); r_eye=get_3d(5)
        l_sh=get_3d(11); r_sh=get_3d(12); l_hip=get_3d(23); r_hip=get_3d(24)
        ear_mid=(l_ear+r_ear)/2; sh_mid=(l_sh+r_sh)/2; hip_mid=(l_hip+r_hip)/2; eye_mid=(l_eye+r_eye)/2
        feats[0] = ear_mid[2] - sh_mid[2]  # HFD
        ear_sh = ear_mid - sh_mid
        vert = np.array([0,-1,0])
        n = np.linalg.norm(ear_sh)
        if n > 1e-6: feats[1] = float(np.degrees(np.arccos(np.clip(np.dot(ear_sh,vert)/n,-1,1))))
        eye_ear = eye_mid - ear_mid
        horiz = np.array([1,0,0])
        n2 = np.linalg.norm(eye_ear)
        if n2 > 1e-6: feats[2] = float(np.degrees(np.arccos(np.clip(np.dot(eye_ear,horiz)/n2,-1,1))))
        trunk = hip_mid - sh_mid
        n3 = np.linalg.norm(trunk)
        if n3 > 1e-6: feats[3] = float(np.degrees(np.arccos(np.clip(np.dot(trunk,vert)/n3,-1,1))))
        sw = np.linalg.norm(r_sh - l_sh)
        if sw > 1e-6: feats[4] = feats[0] / sw
    except: pass
    return feats


# ═══════════════════════════════════════════════════════════════════════
# Universal dataset that loads ALL possible features
# ═══════════════════════════════════════════════════════════════════════
class UniversalNOMDataset(torch.utils.data.Dataset):
    def __init__(self, segment_dict, bone_refs, side_sup, sup_mean, sup_std,
                 stride=TRAIN_STRIDE, augment=False, modality_dropout=0.0):
        self.bone_refs = bone_refs
        self.window_t = WINDOW_T
        self.augment = augment
        self.modality_dropout = modality_dropout
        self.side_sup = side_sup; self.sup_mean = sup_mean; self.sup_std = sup_std
        self.segments = []
        for key, frames in segment_dict.items():
            subj = key[0]
            if subj not in bone_refs: continue
            label = frames[0]["label"]; ref = bone_refs[subj]
            seg_frames = []
            for f in frames:
                w = np.load(f["lm_world"]).astype(np.float32)
                img = np.load(f["lm_img"]).astype(np.float32)
                node = compute_node_features_hybrid(w, img, ref)
                da3_path = f.get("front_da3")
                if da3_path and Path(da3_path).exists():
                    da3 = np.load(da3_path).astype(np.float32)
                    if da3.shape != (224,224): da3 = cv2.resize(da3, (224,224))
                else:
                    da3 = np.zeros((224,224), dtype=np.float32)
                dp_path = DEPTHPRO_ROOT / subj / key[1] / key[2] / f'{f["frame"]}_depth.npy'
                if dp_path.exists():
                    dp = np.load(dp_path).astype(np.float32)
                    if dp.shape != (224,224): dp = cv2.resize(dp, (224,224))
                else:
                    dp = np.zeros((224,224), dtype=np.float32)
                seg_frames.append({"node": node, "world": w, "img": img, "_da3": da3, "_dp": dp,
                                   "subj": subj, "cls": key[1], "dist": key[2], "stem": f["frame"]})
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
        seg = self.segments[si]; frames = seg["frames"][start:start+self.window_t]
        node = np.stack([f["node"] for f in frames], axis=0)
        if self.augment:
            node = node + np.random.randn(*node.shape).astype(np.float32) * 0.005
        mid = self.window_t // 2; mf = frames[mid]
        world_t = np.stack([f["world"] for f in frames], axis=0)
        da3_depth = extract_landmark_depth_40(mf["_da3"], mf["img"])
        da3_norm = extract_landmark_depth_40_normalized(mf["_da3"], mf["img"])
        dp_depth = extract_landmark_depth_40(mf["_dp"], mf["img"])
        ergo = compute_body_frame_30(world_t, mf["_da3"], mf["img"])
        clinical = compute_clinical_3d(mf["img"], mf["_dp"])
        # Mid-frame skeleton for linear probe
        mid_skel = mf["node"].flatten()  # (33*52=1716,) — too big, use mean across joints
        mid_skel_compact = mf["node"].mean(axis=0)  # (52,)
        # Distance feature
        sh_w = float(np.linalg.norm(mf["img"][11] - mf["img"][12]))
        sup_key = (mf["subj"], mf["cls"], mf["dist"], mf["stem"])
        fh_raw = self.side_sup.get(sup_key)
        fh_norm = (fh_raw - self.sup_mean) / self.sup_std if fh_raw is not None else 0.0
        has_sup = 1.0 if fh_raw is not None else 0.0
        return {
            "node": torch.tensor(node, dtype=torch.float32),
            "da3_depth": torch.tensor(da3_depth, dtype=torch.float32),
            "da3_norm": torch.tensor(da3_norm, dtype=torch.float32),
            "dp_depth": torch.tensor(dp_depth, dtype=torch.float32),
            "ergo": torch.tensor(ergo, dtype=torch.float32),
            "clinical_3d": torch.tensor(clinical, dtype=torch.float32),
            "mid_skel": torch.tensor(mid_skel_compact, dtype=torch.float32),
            "sh_width": torch.tensor([sh_w], dtype=torch.float32),
            "dist_code": torch.tensor([0.0], dtype=torch.float32),  # 0=nom
            "fh_dist": torch.tensor(fh_norm, dtype=torch.float32),
            "has_sup": torch.tensor(has_sup, dtype=torch.float32),
            "label": seg["label"],
        }


# ═══════════════════════════════════════════════════════════════════════
# Model factory for each candidate
# ═══════════════════════════════════════════════════════════════════════

def make_model(candidate, hidden=HIDDEN, n_classes=4, dropout=DROPOUT):
    if candidate == 'A':
        # skel+ergo+da3_depth, ECGF gating
        return ModelA(hidden, n_classes, dropout)
    elif candidate == 'B':
        return ModelB(hidden, n_classes, dropout)
    elif candidate == 'C':
        return ModelC(hidden, n_classes, dropout)
    elif candidate == 'D':
        return ModelD(hidden, n_classes, dropout)
    elif candidate == 'E':
        return ModelE(hidden, n_classes, dropout)
    elif candidate == 'F':
        return ModelF(hidden, n_classes, dropout)
    elif candidate == 'G':
        return ModelG(hidden, n_classes, dropout)
    elif candidate == 'H':
        return ModelH(n_classes)
    elif candidate == 'I':
        return ModelI(hidden, n_classes, dropout)
    elif candidate == 'J':
        return ModelJ(hidden, n_classes, dropout)
    elif candidate == 'K':
        return ModelK(hidden, n_classes, dropout)


class ModelA(nn.Module):
    """skel+ergo+da3_depth, ECGF gating"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.ergo = ErgonomicBranch30(in_dim=ERGO_DIM, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.fusion = ErgonomicGatedFusion(n_modalities=3, ergo_dim=ERGO_DIM, hidden=h, dropout=do)
        fd = h + 3*h
        self.cls_head = nn.Sequential(nn.Linear(fd, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); er=self.ergo(batch["ergo"]); dp=self.depth(batch["da3_depth"])
        fused, _=self.fusion([sk,er,dp], batch["ergo"])
        return self.cls_head(fused)

class ModelB(nn.Module):
    """skel only"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        return self.cls_head(self.skeleton(batch["node"]))

class ModelC(nn.Module):
    """skel+ergo, ECGF gating, NO depth"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.ergo = ErgonomicBranch30(in_dim=ERGO_DIM, hidden=h, dropout=do)
        self.fusion = ErgonomicGatedFusion(n_modalities=2, ergo_dim=ERGO_DIM, hidden=h, dropout=do)
        fd = h + 2*h
        self.cls_head = nn.Sequential(nn.Linear(fd, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); er=self.ergo(batch["ergo"])
        fused, _=self.fusion([sk,er], batch["ergo"])
        return self.cls_head(fused)

class ModelD(nn.Module):
    """skel+ergo+da3_depth, simple CONCAT (no ECGF gating)"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.ergo = ErgonomicBranch30(in_dim=ERGO_DIM, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(3*h, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); er=self.ergo(batch["ergo"]); dp=self.depth(batch["da3_depth"])
        return self.cls_head(torch.cat([sk,er,dp], dim=1))

class ModelE(nn.Module):
    """skel + distance-norm depth, NO ergo"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(2*h, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); dp=self.depth(batch["da3_norm"])
        return self.cls_head(torch.cat([sk,dp], dim=1))

class ModelF(nn.Module):
    """skel+ergo+DepthPro depth, simple concat"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.ergo = ErgonomicBranch30(in_dim=ERGO_DIM, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(3*h, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); er=self.ergo(batch["ergo"]); dp=self.depth(batch["dp_depth"])
        return self.cls_head(torch.cat([sk,er,dp], dim=1))

class ModelG(nn.Module):
    """skel + raw 30-dim ergo concat to cls_head (ms_131 style)"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(2*h+ERGO_DIM, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); dp=self.depth(batch["da3_depth"])
        return self.cls_head(torch.cat([sk,dp,batch["ergo"]], dim=1))

class ModelH(nn.Module):
    """Linear probe on mid-frame skeleton (52-dim)"""
    def __init__(self, nc):
        super().__init__()
        self.linear = nn.Linear(52, nc)
    def forward(self, batch, **kw):
        return self.linear(batch["mid_skel"])

class ModelI(nn.Module):
    """skel + DepthPro clinical 3D (CVA3D, HFD, SHA3D)"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.clinical = nn.Sequential(nn.Linear(CLINICAL_3D_DIM, 32), nn.LayerNorm(32), nn.GELU())
        self.cls_head = nn.Sequential(nn.Linear(h+32, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); cl=self.clinical(batch["clinical_3d"])
        return self.cls_head(torch.cat([sk,cl], dim=1))

class ModelJ(nn.Module):
    """skel+ergo+depth+distance_code, simple concat (train on ALL, eval on NOM)"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.ergo = ErgonomicBranch30(in_dim=ERGO_DIM, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.cls_head = nn.Sequential(nn.Linear(3*h+1, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); er=self.ergo(batch["ergo"]); dp=self.depth(batch["da3_depth"])
        return self.cls_head(torch.cat([sk,er,dp,batch["sh_width"]], dim=1))

class ModelK(nn.Module):
    """skel + DepthPro depth + clinical 3D + raw ergo concat"""
    def __init__(self, h, nc, do):
        super().__init__()
        self.skeleton = TemporalSkeletonBranch(n_feat=52, hidden=h, dropout=do)
        self.depth = LandmarkDepthBranch(in_dim=DEPTH_FEAT_DIM, hidden=h, dropout=do)
        self.clinical = nn.Sequential(nn.Linear(CLINICAL_3D_DIM, 32), nn.LayerNorm(32), nn.GELU())
        self.cls_head = nn.Sequential(nn.Linear(2*h+32+ERGO_DIM, h), nn.GELU(), nn.Dropout(do), nn.Linear(h, nc))
    def forward(self, batch, **kw):
        sk=self.skeleton(batch["node"]); dp=self.depth(batch["dp_depth"]); cl=self.clinical(batch["clinical_3d"])
        return self.cls_head(torch.cat([sk,dp,cl,batch["ergo"]], dim=1))


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_fold_universal(candidate, train_ds, val_ds, test_ds, fold_name, seed):
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)
    model = make_model(candidate).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Per-group LR
    param_groups = []
    for name, module in model.named_children():
        lr = 5e-4 if name == "skeleton" else 1e-3
        param_groups.append({"params": module.parameters(), "lr": lr})
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    cls_crit = FocalLoss(gamma=FOCAL_GAMMA, weight=CLASS_WEIGHTS.to(DEVICE))
    best_val, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in train_ld:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(DEVICE) if isinstance(batch["label"], torch.Tensor) \
                else torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)
            logits = model(bd)
            loss = cls_crit(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        val_res = evaluate(model, val_ds)
        if val_res["macro"] > best_val:
            best_val = val_res["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    test_res = evaluate(model, test_ds)
    return best_state, best_val, test_res, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    side_sup, sup_mean, sup_std = load_side_supervision()
    all_segs, rows_per_subj = load_segments_4c_6s()
    nom_segs = {k: v for k, v in all_segs.items() if k[2] == 'nom'}
    bone_refs = {s: compute_subject_bone_reference(rows_per_subj[s])
                 for s in SUBJECTS_6 if s in rows_per_subj}
    anchors = compute_subject_anchors(all_segs, bone_refs, side_sup, sup_mean, sup_std)

    CANDIDATES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    NAMES = {
        'A': 'skel+ergo+DA3, ECGF gate',
        'B': 'skel only',
        'C': 'skel+ergo, ECGF gate',
        'D': 'skel+ergo+DA3, concat',
        'E': 'skel+DA3norm, no ergo',
        'F': 'skel+ergo+DepthPro, concat',
        'G': 'skel+DA3+rawErgo concat',
        'H': 'linear probe (52-dim)',
        'I': 'skel+DepthPro clinical3D',
        'J': 'skel+ergo+DA3+shWidth',
        'K': 'skel+DepthPro+clin3D+rawErgo',
    }

    all_results = {}
    t0 = time.time()

    for cand in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Candidate {cand}: {NAMES[cand]}")
        print(f"{'='*60}")

        # J trains on ALL distances, others on NOM only
        train_segs_source = all_segs if cand == 'J' else nom_segs
        fold_results = []

        for fi, held_out in enumerate(SUBJECTS_6):
            if held_out not in bone_refs: continue
            train_pool = [s for s in SUBJECTS_6 if s != held_out and s in bone_refs]
            val_subj = train_pool[fi % len(train_pool)]
            train_subjects = [s for s in train_pool if s != val_subj]

            train_seg = {k: v for k, v in train_segs_source.items() if k[0] in train_subjects}
            val_seg = {k: v for k, v in nom_segs.items() if k[0] == val_subj}
            test_seg = {k: v for k, v in nom_segs.items() if k[0] == held_out}

            train_ds = AnchoredDataset(UniversalNOMDataset(
                train_seg, bone_refs, side_sup, sup_mean, sup_std,
                stride=TRAIN_STRIDE, augment=True), anchors)
            val_ds = AnchoredDataset(UniversalNOMDataset(
                val_seg, bone_refs, side_sup, sup_mean, sup_std,
                stride=EVAL_STRIDE, augment=False), anchors)
            test_ds = AnchoredDataset(UniversalNOMDataset(
                test_seg, bone_refs, side_sup, sup_mean, sup_std,
                stride=EVAL_STRIDE, augment=False), anchors)

            best_state, best_val, test_res, n_params = train_fold_universal(
                cand, train_ds, val_ds, test_ds,
                f"fold{fi+1}_{held_out}", args.seed + fi)

            pc = test_res['per_class']
            print(f"  [{held_out}] test={test_res['macro']:.1f}%  "
                  f"fh={pc['forward_head']:.0f} L={pc['left_leaning']:.0f} "
                  f"R={pc['right_leaning']:.0f} sit={pc['sit_straight']:.0f}  "
                  f"params={n_params:,d}", flush=True)
            fold_results.append({"held_out": held_out, "test": test_res, "val_macro": best_val})

        # Summary
        m = [f["test"]["macro"] for f in fold_results]
        pc_all = {c: float(np.mean([f["test"]["per_class"][c] for f in fold_results])) for c in CLASSES_4}
        m5 = [f["test"]["macro"] for f in fold_results if f["held_out"] != "pan"]
        pc5 = {c: float(np.mean([f["test"]["per_class"][c] for f in fold_results if f["held_out"] != "pan"])) for c in CLASSES_4}
        a90 = sum(1 for v in pc5.values() if v >= 90)
        a90_6 = sum(1 for v in pc_all.values() if v >= 90)
        print(f"\n  [{cand}] 6-fold: {np.mean(m):.1f}  pc={pc_all}")
        print(f"  [{cand}] 5-subj: {np.mean(m5):.1f}  pc={pc5}  ≥90: {a90}/4")
        all_results[cand] = {"folds": fold_results, "mean_6": float(np.mean(m)),
                              "mean_5": float(np.mean(m5)), "pc5": pc5, "a90_5": a90}

    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON (NOM-only, {(time.time()-t0)/60:.0f} min total)")
    print(f"{'='*70}")
    print(f"{'Cand':>5} {'Name':>35} {'5-subj':>7} {'fh':>6} {'L':>6} {'R':>6} {'sit':>6} {'≥90':>4}")
    for cand in CANDIDATES:
        r = all_results[cand]
        pc = r["pc5"]
        print(f"  {cand:>3}  {NAMES[cand]:>35} {r['mean_5']:>6.1f}  {pc['forward_head']:>5.1f} "
              f"{pc['left_leaning']:>5.1f} {pc['right_leaning']:>5.1f} {pc['sit_straight']:>5.1f}  {r['a90_5']:>3}/4")

    with open(OUT_DIR / "ms160_nom_candidates.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'ms160_nom_candidates.json'}")


if __name__ == "__main__":
    main()
