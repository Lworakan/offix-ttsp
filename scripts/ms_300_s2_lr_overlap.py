"""
Per-feature L vs R separability for a single subject.

Used in §VI to confirm that the residual-failure subject's left and right
lean recordings are not intrinsically ambiguous --- they are highly
separable on the lateral coronal feature shoulder_minus_hip_x, which the
3-D clinical vector f_clin = (sh/ear, HFD, CVA) does not include.

Usage:
    python3 scripts/ms_300_s2_lr_overlap.py [subject]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ms_173_final_9subj import compute_features, load_nom_segments

FEATURE_NAMES = ["sh/ear", "neck_len", "HFD", "CVA", "shoulder_minus_hip_x"]


def per_frame_features(segments, subject, posture):
    key = (subject, posture, "nom")
    if key not in segments:
        return np.empty((0, 5), dtype=np.float32)

    rows = []
    for f in segments[key]:
        world = np.load(f["lm_world"]).astype(np.float32)
        img = np.load(f["lm_img"]).astype(np.float32)
        depth = None
        if f["depthpro"] and Path(f["depthpro"]).exists():
            depth = np.load(f["depthpro"]).astype(np.float32)
            if depth.shape != (224, 224):
                depth = cv2.resize(depth, (224, 224))
        feat4 = compute_features(world, img, depth)  # sh/ear, neck, HFD, CVA
        sh_minus_hip_x = float(((world[11, 0] + world[12, 0]) -
                                 (world[23, 0] + world[24, 0])) / 2)
        rows.append(np.concatenate([feat4, [sh_minus_hip_x]]))
    return np.asarray(rows, dtype=np.float32)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                      (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-9 else float("nan")


def overlap_fraction(a, b, n_grid=200):
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi - lo < 1e-9:
        return 1.0
    grid = np.linspace(lo, hi, n_grid)
    pa, _ = np.histogram(a, bins=grid, density=True)
    pb, _ = np.histogram(b, bins=grid, density=True)
    pa /= max(pa.sum(), 1e-9); pb /= max(pb.sum(), 1e-9)
    return float(1.0 - 0.5 * np.abs(pa - pb).sum())


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "fu"
    segments, _ = load_nom_segments()

    L = per_frame_features(segments, subject, "left_leaning")
    R = per_frame_features(segments, subject, "right_leaning")
    if len(L) == 0 or len(R) == 0:
        sys.exit(f"insufficient data for {subject}: L={len(L)}, R={len(R)}")

    print(f"Subject {subject}  L={len(L)} frames  R={len(R)} frames")
    print(f"{'feature':<22} {'med L':>9} {'med R':>9} {'|Δmed|':>9} "
          f"{'Cohen d':>9} {'overlap':>9} {'p (t)':>10}")
    print("-" * 80)

    summary = []
    for i, name in enumerate(FEATURE_NAMES):
        a, b = L[:, i], R[:, i]
        med_a, med_b = float(np.median(a)), float(np.median(b))
        d = cohens_d(a, b)
        ov = overlap_fraction(a, b)
        try:
            _, p = stats.ttest_ind(a, b, equal_var=False)
        except Exception:
            p = float("nan")
        print(f"{name:<22} {med_a:9.4f} {med_b:9.4f} {abs(med_a-med_b):9.4f} "
              f"{d:9.3f} {ov:9.3f} {p:10.2e}")
        summary.append({"feature": name,
                         "median_left": med_a, "median_right": med_b,
                         "abs_delta_median": abs(med_a - med_b),
                         "cohens_d": d, "overlap": ov, "p_value": float(p)})

    out_path = ROOT / "outputs" / f"lr_overlap_{subject}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "subject": subject,
        "n_left": int(len(L)),
        "n_right": int(len(R)),
        "features": summary,
    }, indent=2, default=float))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
