"""
ms_140_common — Shared dataset loader, feature extractor, and constants for the
new pipeline (ms_141..ms_148) trained on the rebuilt 6-subject dataset.

Single source of truth: data/multisubject/dataset.csv (built by ms_140f)
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "multisubject"
CSV_PATH = DATA / "dataset.csv"
SIDE_SUP_CSV = DATA / "side_supervision.csv"

# 6-subject, 4-class setup
SUBJECTS_6 = ["fu", "mukrop", "nonny", "boom", "peemai", "pan"]
CLASSES_4 = ["forward_head", "left_leaning", "right_leaning", "sit_straight"]
ALL_CLASSES = [
    "forward_head", "freestyle_sitting", "left_leaning",
    "right_leaning", "rounded_shoulders", "sit_straight", "slouched_posture",
]

# Window settings (matching ms_128)
WINDOW_T = 30
TRAIN_STRIDE = 5
EVAL_STRIDE = 15

# Feature dims
ERGO_DIM = 30
DEPTH_FEAT_DIM = 40
DINO_CLS_DIM = 384
DINO_PATCH_DIM = 384
DINO_N_PATCHES = 196  # 14x14 (registers dropped on load)
DINO_REGISTERS = 4

# Side fh_distance is computed/loaded externally
sys.path.insert(0, str(ROOT / "scripts"))
from ms_62_day2_targeted import (
    compute_subject_bone_reference, compute_node_features_hybrid, FocalLoss,
)
from ms_71c_ergonomic_features import (
    compute_ergonomic_window, da3_head_minus_shoulder,
)
from ms_128_ecgf_ldc import (
    extract_landmark_depth_40, compute_body_frame_30,
)


def load_dataset_csv(csv_path: Path = CSV_PATH):
    """Load the master dataset.csv into a list of dict rows."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_side_supervision(csv_path: Path = SIDE_SUP_CSV):
    """Returns dict: (subject, class, distance, frame_stem) -> fh_distance."""
    sup = {}
    if not csv_path.exists():
        return sup, 0.0, 1.0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if int(r["valid"]) == 0:
                continue
            key = (r["subject"], r["class"], r["distance"], r["frame_stem"])
            sup[key] = float(r["fh_distance"])
    vals = np.array(list(sup.values()), dtype=np.float64) if sup else np.array([0.0])
    mean = float(vals.mean())
    std = float(vals.std() + 1e-6)
    return sup, mean, std


def load_segments_4c_6s(rows: Optional[list[dict]] = None, exclude_keys=None):
    """Build per-segment dict over 6 subjects × 4 classes from the new CSV.

    Returns:
      segments: {(subject, class, distance): [list of per-frame dicts]}
      rows_per_subj: {subject: [csv rows for compute_subject_bone_reference]}
    """
    if rows is None:
        rows = load_dataset_csv()
    exclude_keys = set(exclude_keys or [])
    segments = defaultdict(list)
    rows_per_subj = defaultdict(list)
    for r in rows:
        subj = r["subject"]
        cls = r["class"]
        if subj not in SUBJECTS_6: continue
        if cls not in CLASSES_4: continue
        key = (subj, cls, r["distance"])
        if key in exclude_keys: continue
        if not r.get("front_landmarks") or not Path(r["front_landmarks"]).exists():
            continue
        segments[key].append({
            "lm_world": r["front_landmarks"],
            "lm_img": r["front_landmarks_img"],
            "side_lm_world": r.get("side_landmarks", ""),
            "side_lm_img": r.get("side_landmarks_img", ""),
            "front_frame": r.get("front_frame", ""),
            "side_frame": r.get("side_frame", ""),
            "front_da3": r.get("webcam_da3_depth", "") or None,
            "side_da3": r.get("side_da3_depth", "") or None,
            "rs_depth_raw": r.get("rs_depth_raw", "") or None,
            "rs_da3_depth": r.get("rs_da3_depth", "") or None,
            "dinov3_cls": r.get("dinov3_features", "") or None,
            "dinov3_patches": r.get("dinov3_patches", "") or None,
            "label": CLASSES_4.index(cls),
            "subject": subj, "frame": r["frame_stem"],
        })
        # rows_per_subj is used by compute_subject_bone_reference which expects
        # the legacy row format with `webcam_landmarks` field. Map it.
        rows_per_subj[subj].append({
            "subject": subj, "class": cls, "distance": r["distance"],
            "frame_stem": r["frame_stem"],
            "webcam_landmarks": r["front_landmarks"],
        })
    for k in segments:
        segments[k].sort(key=lambda f: f["frame"])
    return segments, rows_per_subj


class WindowedDatasetV2(Dataset):
    """New dataset class for ms_141..ms_148.

    Outputs per window:
      node:            (T, 33, 52) hybrid skeleton features (front)
      landmark_depth:  (40,)
      ergo:            (30,) body-frame clinical features
      dino_cls:        (384,) DINOv3 CLS pooler (front)
      dino_patches:    (T, 196, 384) DINOv3 patch tokens (front, optional)
      side_lm:         (T, 33, 2) side image landmarks (privileged, training only)
      fh_dist:         scalar normalized side fh_distance (privileged label)
      has_sup:         scalar 1.0 if side supervision available
      label:           int

    Toggle features via include_* flags to support the ablation chain
    (skel-only → +ergo → +depth → +dinoCLS → +dinoPatches → +sideLM).
    """
    def __init__(self, segment_dict, bone_refs, side_sup, sup_mean, sup_std,
                 window_t=WINDOW_T, stride=TRAIN_STRIDE, augment=False,
                 modality_dropout=0.0,
                 include_landmark_depth=True,
                 include_ergo=True,
                 include_dino_cls=True,
                 include_dino_patches=False,
                 include_side_lm=False,
                 include_side_depth=False):
        self.bone_refs = bone_refs
        self.window_t = window_t
        self.augment = augment
        self.modality_dropout = modality_dropout
        self.side_sup = side_sup
        self.sup_mean = sup_mean
        self.sup_std = sup_std
        self.inc_ldepth = include_landmark_depth
        self.inc_ergo = include_ergo
        self.inc_dino_cls = include_dino_cls
        self.inc_dino_patches = include_dino_patches
        self.inc_side_lm = include_side_lm
        self.inc_side_depth = include_side_depth
        self.segments = []

        for key, frames in segment_dict.items():
            subj = key[0]; cls_name = key[1]; dist = key[2]
            if subj not in bone_refs: continue
            label = frames[0]["label"]
            ref = bone_refs[subj]
            seg_frames = []
            for f in frames:
                world = np.load(f["lm_world"]).astype(np.float32)
                img_lm = np.load(f["lm_img"]).astype(np.float32)
                node = compute_node_features_hybrid(world, img_lm, ref)

                # Front DA3 (always loaded for ergo + landmark depth)
                if f["front_da3"] and Path(f["front_da3"]).exists():
                    da3 = np.load(f["front_da3"]).astype(np.float32)
                    if da3.shape != (224, 224):
                        da3 = cv2.resize(da3, (224, 224), interpolation=cv2.INTER_LINEAR)
                else:
                    da3 = np.zeros((224, 224), dtype=np.float32)

                # DINOv3 CLS (optional)
                dino_cls = np.zeros(DINO_CLS_DIM, dtype=np.float32)
                if self.inc_dino_cls and f.get("dinov3_cls") and Path(f["dinov3_cls"]).exists():
                    dino_cls = np.load(f["dinov3_cls"]).astype(np.float32)

                # DINOv3 patches (optional)
                dino_patches = np.zeros((DINO_N_PATCHES, DINO_PATCH_DIM), dtype=np.float32)
                if self.inc_dino_patches and f.get("dinov3_patches") and Path(f["dinov3_patches"]).exists():
                    raw = np.load(f["dinov3_patches"]).astype(np.float32)  # (200, 384) [4 reg + 196 patches]
                    dino_patches = raw[DINO_REGISTERS:]

                # Side image landmarks (privileged, optional)
                side_lm = np.zeros((33, 2), dtype=np.float32)
                if self.inc_side_lm and f.get("side_lm_img") and Path(f["side_lm_img"]).exists():
                    raw_side = np.load(f["side_lm_img"]).astype(np.float32)
                    if raw_side.shape == (33, 2):
                        side_lm = raw_side
                    elif raw_side.shape == (33, 3):
                        side_lm = raw_side[:, :2]

                # Side DA3 depth — privileged training-only encoder input
                side_da3 = np.zeros((224, 224), dtype=np.float32)
                if self.inc_side_depth and f.get("side_da3") and Path(f["side_da3"]).exists():
                    sd = np.load(f["side_da3"]).astype(np.float32)
                    if sd.shape != (224, 224):
                        sd = cv2.resize(sd, (224, 224), interpolation=cv2.INTER_LINEAR)
                    side_da3 = sd

                seg_frames.append({
                    "node": node, "world": world, "img": img_lm,
                    "_da3": da3, "dino_cls": dino_cls, "dino_patches": dino_patches,
                    "side_lm": side_lm, "_side_da3": side_da3,
                    "subj": subj, "cls": cls_name, "dist": dist,
                    "stem": f["frame"],
                })
            self.segments.append({"subject": subj, "label": label,
                                   "cls_name": cls_name, "dist": dist,
                                   "frames": seg_frames})
        self.windows = []
        for si, seg in enumerate(self.segments):
            n = len(seg["frames"])
            if n < window_t: continue
            for start in range(0, n - window_t + 1, stride):
                self.windows.append((si, start))

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        si, start = self.windows[idx]
        seg = self.segments[si]
        frames = seg["frames"][start:start + self.window_t]

        node = np.stack([f["node"] for f in frames], axis=0)
        if self.augment:
            node = node + np.random.randn(*node.shape).astype(np.float32) * 0.005
            if np.random.rand() < 0.5:
                sx = np.random.uniform(0.92, 1.08); sy = np.random.uniform(0.92, 1.08)
                node[..., 0] *= sx; node[..., 1] *= sy
                node[..., 26] *= sx; node[..., 27] *= sy

        mid = self.window_t // 2
        mf = frames[mid]
        world_t = np.stack([f["world"] for f in frames], axis=0)
        img_lm_mid = mf["img"]
        da3_mid = mf["_da3"]

        # Landmark-aligned depth (40-dim)
        ldepth = extract_landmark_depth_40(da3_mid, img_lm_mid) if self.inc_ldepth else np.zeros(40, dtype=np.float32)
        # 30-dim ergo
        ergo = compute_body_frame_30(world_t, da3_mid, img_lm_mid) if self.inc_ergo else np.zeros(30, dtype=np.float32)
        # DINOv3 CLS — temporal mean across window
        if self.inc_dino_cls:
            dino_cls = np.mean([f["dino_cls"] for f in frames], axis=0).astype(np.float32)
        else:
            dino_cls = np.zeros(DINO_CLS_DIM, dtype=np.float32)
        # DINOv3 patches per frame
        if self.inc_dino_patches:
            dino_patches = np.stack([f["dino_patches"] for f in frames], axis=0).astype(np.float32)
        else:
            dino_patches = np.zeros((self.window_t, DINO_N_PATCHES, DINO_PATCH_DIM), dtype=np.float32)
        # Side image landmarks per frame (privileged, training only, optional)
        if self.inc_side_lm:
            side_lm = np.stack([f["side_lm"] for f in frames], axis=0).astype(np.float32)
        else:
            side_lm = np.zeros((self.window_t, 33, 2), dtype=np.float32)

        # Side DA3 depth at middle frame (privileged training-only encoder input)
        if self.inc_side_depth:
            side_da3_mid = mf["_side_da3"]
        else:
            side_da3_mid = np.zeros((224, 224), dtype=np.float32)

        if self.augment and self.modality_dropout > 0:
            r = np.random.rand()
            if r < self.modality_dropout / 3:
                ldepth = np.zeros_like(ldepth)
            elif r < 2 * self.modality_dropout / 3:
                dino_cls = np.zeros_like(dino_cls)
                dino_patches = np.zeros_like(dino_patches)

        # Side fh_distance lookup
        sup_key = (mf["subj"], mf["cls"], mf["dist"], mf["stem"])
        fh_raw = self.side_sup.get(sup_key)
        if fh_raw is not None:
            fh_norm = (fh_raw - self.sup_mean) / self.sup_std
            has_sup = 1.0
        else:
            fh_norm = 0.0; has_sup = 0.0

        return {
            "node": torch.tensor(node, dtype=torch.float32),
            "landmark_depth": torch.tensor(ldepth, dtype=torch.float32),
            "ergo": torch.tensor(ergo, dtype=torch.float32),
            "dino_cls": torch.tensor(dino_cls, dtype=torch.float32),
            "dino_patches": torch.tensor(dino_patches, dtype=torch.float32),
            "side_lm": torch.tensor(side_lm, dtype=torch.float32),
            "side_depth": torch.tensor(side_da3_mid, dtype=torch.float32),
            "fh_dist": torch.tensor(fh_norm, dtype=torch.float32),
            "has_sup": torch.tensor(has_sup, dtype=torch.float32),
            "label": seg["label"],
        }


def evaluate(model, ds, batch_size=64, forward_kwargs=None):
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    forward_kwargs = forward_kwargs or {}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
            logits = model(bd, **forward_kwargs)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["label"].numpy() if isinstance(batch["label"], torch.Tensor) \
                else np.array(batch["label"])
            all_preds.append(preds); all_labels.append(labels)
    preds = np.concatenate(all_preds); labels = np.concatenate(all_labels)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES_4))))
    pc = {}
    for i, c in enumerate(CLASSES_4):
        n = cm[i].sum()
        pc[c] = float(cm[i, i] / n * 100) if n > 0 else 0.0
    macro = float(np.mean(list(pc.values())))
    return {"macro": macro, "per_class": pc, "acc": float((preds == labels).mean() * 100), "cm": cm.tolist()}
