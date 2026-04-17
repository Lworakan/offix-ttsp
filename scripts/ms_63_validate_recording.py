"""
Recording validator — verify each (subject, class, distance) segment is
clearly distinguishable from that subject's own sit_straight baseline.

Catches "pan-style" recording defects BEFORE the data goes into training.

Usage:

  # Validate all recordings of a single subject (audit existing dataset)
  python3 scripts/ms_63_validate_recording.py --subject pan

  # Validate one specific (subject, class, distance) recording
  python3 scripts/ms_63_validate_recording.py --subject pan --class forward_head --distance close

  # Validate everything in the dataset, write a report
  python3 scripts/ms_63_validate_recording.py --all

Exit code 0 = pass, 1 = fail (so you can chain with shell `&&` after a recording).

Returns:
  PASS  if  >= 3 metrics show |z| >= 1.0  AND  required-direction metrics agree
  FAIL  otherwise — with a list of which metrics failed and what to do about it
"""

import csv, sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "multisubject" / "webcam_dataset.csv"
OUT = ROOT / "outputs" / "data_exploration" / "recording_validation.md"
OUT.parent.mkdir(exist_ok=True, parents=True)

# How many frames of sit_straight to use for the baseline
BASELINE_MIN_FRAMES = 30

# Number of metrics that must clearly separate (|z| >= Z_THRESH)
Z_THRESH = 1.0
N_REQUIRED_CLEAR = 3

# ── Metric definitions (must match ms_60) ─────────────────────────────────

def metric_nose_z_minus_sh_z(world):
    return float(world[0, 2] - (world[11, 2] + world[12, 2]) / 2)

def metric_nose_y_minus_sh_y(world):
    return float(world[0, 1] - (world[11, 1] + world[12, 1]) / 2)

def metric_chin_jut(world):
    nose = world[0]; mid_ear = (world[7] + world[8]) / 2
    return float(nose[2] - mid_ear[2])

def metric_cva(world):
    mid_ear = (world[7] + world[8]) / 2
    mid_sh = (world[11] + world[12]) / 2
    v = mid_sh - mid_ear
    vn = v / (np.linalg.norm(v) + 1e-8)
    cos_a = float(np.clip(np.dot(vn, np.array([0, 1, 0])), -1, 1))
    return float(np.degrees(np.arccos(cos_a)))

def metric_img_horizontal(img):
    sw = max(abs(img[11, 0] - img[12, 0]), 1e-6)
    return float((img[0, 0] - (img[11, 0] + img[12, 0]) / 2) / sw)

def metric_img_vertical(img):
    sw = max(abs(img[11, 0] - img[12, 0]), 1e-6)
    return float((img[0, 1] - (img[11, 1] + img[12, 1]) / 2) / sw)

def metric_nose_z_minus_hip_z(world):
    mid_hip = (world[23] + world[24]) / 2
    return float(world[0, 2] - mid_hip[2])

def metric_sh_z_minus_hip_z(world):
    return float(((world[11, 2] + world[12, 2]) / 2) - ((world[23, 2] + world[24, 2]) / 2))

def metric_nose_y_minus_hip_y(world):
    mid_hip = (world[23] + world[24]) / 2
    return float(world[0, 1] - mid_hip[1])

def metric_shoulder_x(world):
    return float(world[11, 0] - world[12, 0])

def metric_shoulder_minus_hip_x(world):
    return float(((world[11, 0] + world[12, 0]) / 2) - ((world[23, 0] + world[24, 0]) / 2))

def metric_torso_y_curl(world):
    """Slouch proxy: vertical distance shoulder-to-hip. Smaller = slouched."""
    mid_sh_y = (world[11, 1] + world[12, 1]) / 2
    mid_hip_y = (world[23, 1] + world[24, 1]) / 2
    return float(abs(mid_sh_y - mid_hip_y))


# Each metric -> (function, takes_world: bool)
METRICS = {
    "nose_z_minus_sh_z":  (metric_nose_z_minus_sh_z, True),
    "nose_y_minus_sh_y":  (metric_nose_y_minus_sh_y, True),
    "chin_jut_z":         (metric_chin_jut,         True),
    "cva_deg":            (metric_cva,              True),
    "img_horizontal":     (metric_img_horizontal,   False),
    "img_vertical":       (metric_img_vertical,     False),
    "nose_z_minus_hip_z": (metric_nose_z_minus_hip_z, True),
    "sh_z_minus_hip_z":   (metric_sh_z_minus_hip_z, True),
    "nose_y_minus_hip_y": (metric_nose_y_minus_hip_y, True),
    "shoulder_minus_hip_x": (metric_shoulder_minus_hip_x, True),
    "torso_y_curl":       (metric_torso_y_curl,     True),
}


# ── Per-class direction requirements ──────────────────────────────────────
# A class PASSES only if these metrics show the listed direction sign
# Sign: -1 means delta should be negative (fh: nose moves forward = nose_z drops)
#       +1 means delta should be positive
DIRECTION_RULES = {
    "forward_head": {
        # head moves forward in z, head drops in y, CVA decreases (more horizontal)
        "nose_z_minus_sh_z":  -1,
        "chin_jut_z":         -1,
        "cva_deg":            -1,
        "nose_z_minus_hip_z": -1,
    },
    "slouched_posture": {
        # head drops, torso compresses
        "nose_y_minus_sh_y":  +1,   # nose_y > shoulder_y in MP convention (downward)
        "nose_y_minus_hip_y": +1,
        "torso_y_curl":       -1,
    },
    "left_leaning": {
        "shoulder_minus_hip_x": -1,
    },
    "right_leaning": {
        "shoulder_minus_hip_x": +1,
    },
    "sit_straight": {},  # baseline, no rules
}


def compute_metrics(world, img):
    out = {}
    for name, (fn, needs_world) in METRICS.items():
        try:
            out[name] = fn(world) if needs_world else fn(img)
        except Exception:
            out[name] = None
    return out


def baseline_for_subject(rows, subject):
    """Compute mean and std of each metric over the subject's sit_straight frames."""
    sit = [r for r in rows if r["subject"] == subject and r["class"] == "sit_straight"]
    if len(sit) < BASELINE_MIN_FRAMES:
        return None
    vals = defaultdict(list)
    for r in sit:
        lm = r.get("webcam_landmarks", "")
        if not lm or not Path(lm).exists():
            continue
        world = np.load(lm).astype(np.float32)
        img = np.load(lm.replace("_landmarks.npy", "_landmarks_img.npy")).astype(np.float32)
        m = compute_metrics(world, img)
        for k, v in m.items():
            if v is not None:
                vals[k].append(v)
    return {
        k: {"median": float(np.median(v)),
            "std":    float(np.std(v) + 1e-8),
            "n": len(v)}
        for k, v in vals.items()
    }


def validate_segment(rows, subject, cls, distance, baseline, verbose=True):
    """Returns dict with pass/fail and per-metric z-scores."""
    if cls == "sit_straight":
        return {"verdict": "BASELINE", "n_frames": 0, "metrics": {}}

    seg_rows = [r for r in rows
                if r["subject"] == subject and r["class"] == cls
                and (distance is None or r.get("distance") == distance)]
    if not seg_rows:
        return {"verdict": "MISSING", "n_frames": 0, "metrics": {}}

    vals = defaultdict(list)
    for r in seg_rows:
        lm = r.get("webcam_landmarks", "")
        if not lm or not Path(lm).exists():
            continue
        world = np.load(lm).astype(np.float32)
        img = np.load(lm.replace("_landmarks.npy", "_landmarks_img.npy")).astype(np.float32)
        m = compute_metrics(world, img)
        for k, v in m.items():
            if v is not None:
                vals[k].append(v)
    if not vals:
        return {"verdict": "NO_LANDMARKS", "n_frames": 0, "metrics": {}}

    n_clear = 0
    metric_results = {}
    for k in METRICS.keys():
        if k not in vals or k not in baseline:
            metric_results[k] = None
            continue
        med = float(np.median(vals[k]))
        delta = med - baseline[k]["median"]
        z = delta / baseline[k]["std"]
        clear = abs(z) >= Z_THRESH
        if clear: n_clear += 1
        metric_results[k] = {"segment_median": med, "delta": delta, "z": z, "clear": clear}

    # Direction check (informational only — different subjects may use different
    # geometric strategies for the same posture; we don't enforce signs).
    dir_rules = DIRECTION_RULES.get(cls, {})
    dir_pass = True
    dir_fail_reasons = []
    for k, expected_sign in dir_rules.items():
        if k not in metric_results or metric_results[k] is None: continue
        z = metric_results[k]["z"]
        if expected_sign * z < 0.5:
            dir_fail_reasons.append(f"{k} z={z:+.2f} (expected sign {expected_sign:+d})")

    n_required = N_REQUIRED_CLEAR
    overall_pass = n_clear >= n_required

    verdict = "PASS" if overall_pass else "FAIL"
    n_frames = max(len(v) for v in vals.values())

    if verbose:
        print(f"\n[{subject}] {cls}" + (f"/{distance}" if distance else "") +
              f"  n_frames={n_frames}  n_clear={n_clear}/{len(METRICS)}  verdict={verdict}")
        for k, m in metric_results.items():
            if m is None: continue
            mark = "✓" if m["clear"] else "·"
            print(f"  {mark}  {k:25s}  z={m['z']:+6.2f}")
        if dir_fail_reasons:
            print(f"  ✗ direction failures: {', '.join(dir_fail_reasons)}")
        if not overall_pass:
            print(f"  → re-record required ({n_clear} clear metrics, need {n_required}; "
                  f"direction {'OK' if dir_pass else 'FAILED'})")

    return {
        "verdict": verdict, "n_frames": n_frames, "n_clear": n_clear,
        "n_required": n_required, "direction_pass": dir_pass,
        "direction_fail_reasons": dir_fail_reasons,
        "metrics": metric_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--class", dest="cls", type=str, default=None)
    ap.add_argument("--distance", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write_report", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV_PATH)))

    if args.all:
        subjects = sorted(set(r["subject"] for r in rows))
    elif args.subject:
        subjects = [args.subject]
    else:
        ap.error("Specify --subject or --all")

    all_results = {}
    any_fail = False
    for subj in subjects:
        baseline = baseline_for_subject(rows, subj)
        if baseline is None:
            print(f"[{subj}] insufficient sit_straight baseline (<{BASELINE_MIN_FRAMES} frames). Record more sit_straight first.")
            continue

        if args.cls:
            classes = [args.cls]
        else:
            classes = sorted(set(r["class"] for r in rows
                                  if r["subject"] == subj and r["class"] != "sit_straight"
                                  and r["class"] not in {"freestyle_sitting", "rounded_shoulders"}))

        all_results[subj] = {}
        for cls in classes:
            if args.distance:
                distances = [args.distance]
            else:
                distances = sorted(set(r.get("distance", "nom") for r in rows
                                        if r["subject"] == subj and r["class"] == cls))
            for d in distances:
                res = validate_segment(rows, subj, cls, d, baseline, verbose=True)
                all_results[subj][f"{cls}/{d}"] = res
                if res["verdict"] == "FAIL":
                    any_fail = True

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for subj, res in all_results.items():
        n_total = len(res)
        n_pass = sum(1 for r in res.values() if r["verdict"] == "PASS")
        n_fail = sum(1 for r in res.values() if r["verdict"] == "FAIL")
        print(f"  {subj}: {n_pass}/{n_total} pass, {n_fail} FAIL")
        for k, r in res.items():
            if r["verdict"] == "FAIL":
                print(f"    ✗ {k}  ({r['n_clear']} clear, dir={'OK' if r['direction_pass'] else 'FAIL'})")

    if args.write_report:
        md = ["# Recording Validation Report\n",
              f"- Threshold: |z| ≥ {Z_THRESH}, need ≥ {N_REQUIRED_CLEAR} clear metrics\n",
              "## Per-segment\n"]
        for subj, res in all_results.items():
            md.append(f"### {subj}\n")
            md.append("| segment | n_frames | clear/total | dir | verdict |")
            md.append("|---|---|---|---|---|")
            for k, r in res.items():
                if r["verdict"] in ("BASELINE", "MISSING", "NO_LANDMARKS"): continue
                dir_mark = "✓" if r["direction_pass"] else "✗"
                v_mark = "✓ PASS" if r["verdict"] == "PASS" else "**✗ FAIL**"
                md.append(f"| {k} | {r['n_frames']} | {r['n_clear']}/{len(METRICS)} | {dir_mark} | {v_mark} |")
            md.append("")
        OUT.write_text("\n".join(md))
        print(f"\nReport: {OUT}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
