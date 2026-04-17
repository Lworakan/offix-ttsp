# TTSP Deployment-Time Calibration Protocol

This document specifies the **per-user calibration step** that a new user must
complete the first time they open the TTSP webcam demo at
[https://offix.tech](https://offix.tech). It is the source of the per-subject
anchors $\mathbf{a}_{LR}$ and $\mathbf{a}_{SIT}$ used by the dual-expert
classifier described in the CVIPPR 2026 paper (`paper/cvippr_2026/main.tex`).

> **Why this exists.** TTSP is *gradient-free at inference* (the model never
> back-propagates after deployment), but it is **not** *zero-label*: the user
> must provide a brief few-shot calibration recording in which they tell the
> system which posture they are currently performing.

---

## What the user does (~10 seconds total)

A wizard collects three short clips, each of which the user must self-label
by clicking the corresponding button before they hold the pose:

| step | posture | duration | frames @ 6 fps |
|------|---------|----------|----------------|
| 1    | left-lean   | ≈ 4 s | 25 |
| 2    | right-lean  | ≈ 4 s | 25 |
| 3    | sit-upright | ≈ 4 s | 25 |

Per posture, the user is shown the same on-screen skeleton overlay used at
data-collection time (`scripts/ms_67_record_session.py`) with the standard
ergonomic cues from `data/multisubject/RECORDING_PROTOCOL.md` §The 5 postures
(strings-from-crown for sit-upright, drop-the-corresponding-shoulder for the
leans). The wizard advances when MediaPipe Pose has produced 25 valid
frames; no z-score or PASS/FAIL gating is applied at deployment time —
unlike data collection, the user is not retried if their pose is mild.

## How the anchors are computed from these frames

Implemented in:

- `scripts/ms_173_final_9subj.py` → `compute_lr_anchors()` (lines 108–126)
- `scripts/ms_204_sit_anchor.py` → `compute_sit_anchors()` (lines 53–70)

Both functions:

1. Extract MediaPipe world + image landmarks per frame from `webcam_landmarks/`.
2. Read the corresponding DepthPro depth map from `depthpro/`.
3. Call `compute_features()` (in `ms_173_final_9subj.py`, lines 89–105) to
   produce a per-frame $\mathbf{f}_{clin}\!\in\!\mathbb{R}^{4}$ vector
   (sh/ear ratio, HFD, CVA, shoulder–hip-x).
4. Take the per-subject **mean** over the 25 calibration frames (50 for
   $\mathbf{a}_{LR}$, since left + right are pooled).

The resulting per-subject 4-vectors are the only labelled artefacts the
deployed system ever consumes.

## What happens at inference (no further user action)

For each 7-frame webcam window, both experts (LR-anchored and SIT-anchored)
produce a softmax $P_{LR}, P_{SIT}\!\in\!\mathbb{R}^{4}$. The TTSP entropy
gate computes $\bar H(P_{LR})$ and $\bar H(P_{SIT})$ over the recent
window-buffer and routes:

```
g = 1 if mean_entropy(P_LR) < mean_entropy(P_SIT) else 0
P_TTSP = g * P_LR + (1 - g) * P_SIT
y_hat  = argmax(P_TTSP)
```

The gate has no learned parameters. The model has no gradient updates after
calibration. The calibration anchors are stored client-side (browser
LocalStorage / equivalent on-device) and are never uploaded to a server.

---

## Honest characterisation of the deployment regime

| dimension | TTSP |
|-----------|------|
| Labelled samples at training time | yes (full LOSO supervision on N−1 subjects) |
| Labelled samples at deployment time | yes — three short self-labelled calibration clips |
| Gradient updates after deployment | **no** |
| Re-training when a new user arrives | **no** |
| Cloud uploads at inference | **no** (ONNX-Web, on-device) |

The accurate one-line description is therefore **"few-shot calibrated,
gradient-free at inference"**, not "zero-label".
