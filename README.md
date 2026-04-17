# TTSP: Test-Time Subject Probing for Cross-Subject Sitting-Posture Classification

Reference implementation and paper source for the CVIPPR 2026 submission
*TTSP: Test-Time Subject Probing for Cross-Subject Sitting-Posture Classification
from a Single Front Webcam*.

The contribution is a calibration-validated method selector that routes each
user to one of four deployable inference heads (skeleton-only baseline,
LR-anchored expert with class-protected offset head, entropy-gated dual-expert
variant, 1-shot prototype classifier). On a strict 9-subject LOSO benchmark
the selector reaches the per-subject oracle upper bound exactly  *99.2 %*
cohort macro-F1, 8/9 subjects at >= 90 % per-class accuracy while using
only the user's own labelled calibration recording at deployment time.

## Repository layout

```
paper/cvippr_2026/       LaTeX source + compiled PDF
scripts/                 training, evaluation, and analysis scripts
outputs/                 per-sample predictions + aggregated JSON results
data/multisubject/       dataset metadata (recording protocol, demographics)
CALIBRATION_PROTOCOL.md  deployment-time wizard specification
requirements.txt
```

Raw recordings, landmarks, and depth maps are held out of the public repo
for subject privacy and file-size reasons; the scripts assume the
multi-subject dataset lives under
`data/multisubject/{webcam_frames, webcam_landmarks, depthpro}/<subject>/<class>/<distance>/`
and can be populated by following
`data/multisubject/RECORDING_PROTOCOL.md`.

## Reproducing the paper

1. Install dependencies (Python 3.10):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the three main experiments (each writes per-sample predictions under
   `outputs/`):

   ```bash
   python3 scripts/ms_173_final_9subj.py --seed 42 --only E1a --save_preds
   python3 scripts/ms_225_ttsp.py        --seed 42
   python3 scripts/ms_224_oneshot_v2.py  --seed 42 --shots 1
   ```

3. Derive the headline numbers and Table I statistics:

   ```bash
   python3 scripts/ms_227_honest_selector.py       # honest TTSP-selector
   python3 scripts/ms_302_table1_stats.py          # bootstrap 95% CI + McNemar
   python3 scripts/ms_300_s2_lr_overlap.py mukrop  # S3 feature-overlap table
   ```

4. Build the paper:

   ```bash
   cd paper/cvippr_2026
   pdflatex main && bibtex main && pdflatex main && pdflatex main
   ```

## License

Code and paper source are released for academic use. Please cite the
CVIPPR 2026 paper if you use this work.

## Contact

See the author block of the paper for corresponding-author emails.
