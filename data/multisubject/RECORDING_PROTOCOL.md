# Sitting Posture Recording Protocol

This is a one-pager protocol for recording sitting posture data for the
SitPose-CIPCV 2026 study. Total time per subject: **15–20 minutes**.

You will record yourself in **5 sitting postures × 3 distances = 15 segments**,
each segment ~30 seconds. The on-screen tool validates each recording in real
time and won't save invalid poses.

---

## Equipment

- Laptop with webcam (any 720p+ camera)
- A chair without armrests (or with low armrests that don't block your torso)
- A flat desk or table at typical height
- A measuring tape or ruler (or use your forearm — see below)
- ~5 minutes of quiet space without distractions

## Setup (one-time, ~5 min)

1. **Position the laptop** on the desk in front of you.
2. **Camera at chin level**: when you sit upright, the webcam should look at your
   chin/mouth area. If too low, place a book under the laptop. If too high,
   stack books in front of it.
3. **Plain background** behind you. White wall is best. Avoid windows behind
   you (backlight).
4. **Decent lighting** — daylight or a desk lamp facing you. Don't sit in shadow.
5. **Wear normal clothes** — t-shirt or fitted top is best. Avoid loose
   sweaters that hide your shoulders. No scarves.

## The 3 distances

| name | camera-to-chest distance | how to measure |
|---|---|---|
| **close** | ~50 cm | one of your forearms (elbow to fingertip) from desk edge |
| **nom**   | ~70 cm | comfortable laptop typing distance |
| **far**   | ~90 cm | sit back, arms slightly extended toward keyboard |

Mark each chair position with **a piece of tape on the floor** so you can return to it.

## The 5 postures

### 1. sit_straight (REFERENCE — record this FIRST)
- Sit upright on your sit-bones
- Ears directly above shoulders
- Shoulders directly above hips
- Look straight ahead at the laptop
- Both feet flat on the floor
- Hold a relaxed neutral position
- **Mental cue**: "as if a string is gently pulling the top of your head up"

### 2. forward_head
- Keep your back the same as sit_straight
- **Push your head forward** so your nose is in front of your shoulder line
- Imagine you're trying to read small text on the screen and you lean your head closer
- The chin should jut forward, not down
- Hold so your CVA bar shows green

### 3. slouched_posture
- Round your upper back forward (thoracic kyphosis)
- Let your shoulders drop forward and inward
- Let your head fall slightly down
- Feel your stomach compress slightly
- This is the "tired at desk" pose

### 4. left_leaning
- Lean your **whole upper body** to the left at the waist
- Drop your **left** shoulder noticeably lower than the right
- Keep your hips on the chair (don't slide)
- Don't tilt your head independently — the head goes with the torso

### 5. right_leaning
- Mirror image of left_leaning
- Lean to the right, drop the right shoulder lower

---

## Recording procedure (per posture × distance = 1 segment)

For each of the 15 segments:

1. **Move to the right distance** (use your tape mark)
2. **Open a terminal in the project directory**:
   ```bash
   cd ~/Documents/GitHub/offix
   python3 scripts/ms_67_record_session.py \
     --subject YOUR_NAME --class POSTURE_NAME --distance DISTANCE_NAME
   ```
   Replace `YOUR_NAME`, `POSTURE_NAME` (sit_straight / forward_head / slouched_posture
   / left_leaning / right_leaning), `DISTANCE_NAME` (close / nom / far).
3. The webcam window opens. You will see your skeleton overlay and a panel
   on the right showing **11 live ergonomic metrics** with z-score bars.
4. **Get into the posture slowly**, watching the bars on the right of the screen:
   - Bars turn **green** when the metric reaches |z| ≥ 1.0 (clear separation)
   - You need **at least 3 green bars** at the same time to enter PASS state
   - The status bar at top shows **PASS** when ready
5. **Press SPACE** to start recording. Recording only fills while you are in PASS
   state — if you drift back, recording silently pauses.
6. **Hold the pose**, watching the bars. Try to push for 4–5 green bars, not
   just the bare minimum.
7. When the buffer reaches the target (~180 frames = 30 seconds), the screen
   shows "TARGET REACHED — press S to save".
8. Press **S**. Frames and landmarks are saved to disk.

### Validate immediately after each segment

```bash
python3 scripts/ms_63_validate_recording.py \
  --subject YOUR_NAME --class POSTURE_NAME --distance DISTANCE_NAME
```

- **PASS** → keep, move on to next segment
- **FAIL** → press R to reset, re-record more strongly

**Don't skip validation.** It takes 1 second and catches invalid recordings on
the spot, before you continue with later postures.

---

## Recommended order

To minimize distance changes (which take time), record all 5 postures at one
distance before moving to the next:

1. Start at **close** distance:
   - sit_straight → forward_head → slouched_posture → left_leaning → right_leaning
2. Move to **nom** distance, repeat all 5 postures
3. Move to **far** distance, repeat all 5 postures

Total time per subject:
- 15 segments × ~1 min each (including setup + validation) = **~15 min**
- Plus 5 min initial setup = **~20 min total**

---

## Common mistakes that cause segment failure

| Mistake | Symptom | How to fix |
|---|---|---|
| Posture too mild | Bars stay grey or yellow | Push the posture more strongly. forward_head needs visible head shift. |
| Drifting back to neutral | Some frames PASS, some don't | Brace your back/hips against the chair so the targeted body part holds steady |
| Wrong joint moving | e.g., tilting head down instead of forward | Read the posture description again — make sure the right body part is moving |
| Sitting too far/close from camera | "NO POSE" or partial detection | Adjust laptop position or add books underneath until your head is ~25% of the frame |
| Loose clothing | Shoulder asymmetry bars look strange | Wear a fitted top so shoulder positions are visible |
| Poor lighting | MediaPipe loses your face | Add a desk lamp facing you |

---

## When to give up and move on

If after **5 attempts** you cannot reach PASS for a particular distance:

1. Check that your sit_straight baseline is correct (run the validator on
   sit_straight first; if it's weird, re-record sit_straight)
2. Try the next distance (some subjects fail at one distance but pass at others)
3. Ask another person to evaluate the pose visually — sometimes you think
   you're posing but the camera disagrees

Each segment that you can't record is one piece of data the model loses.
Better to take an extra minute and get it right than to skip and have a weak
recording that hurts the model.

---

## Final audit (after all 15 segments)

```bash
python3 scripts/ms_63_validate_recording.py --subject YOUR_NAME --write_report
```

This walks all your saved segments and prints a final pass/fail summary.
**You should see all 15 segments pass.** If any fail, re-record those before
sending the data.

---

## What gets saved

The tool saves to:

```
data/multisubject/
├── webcam_frames/YOUR_NAME/<posture>/<distance>/0000.png ... 0179.png
└── webcam_landmarks/YOUR_NAME/<posture>/<distance>/0000_landmarks.npy ... 0179_landmarks_img.npy
```

Each segment is ~180 frames at 6 fps = 30 seconds.

After all 5 subjects record, run the dataset validator and the depth/feature
extractors:

```bash
python3 scripts/ms_63_validate_recording.py --all --write_report
python3 scripts/ms_72b_dataset_validation.py
```

---

## Privacy

The recordings contain your face, body, and background of where you're sitting.
For the paper we will:
- Pixelate or crop faces in any published figures
- Release only MediaPipe skeleton .npy files (no raw images) on request
- Anonymize your name to S1–S6 in the paper

If you are uncomfortable with any of this, tell me before recording.

---

## Quick reference card

```
Setup:
  1. Camera at chin level
  2. Plain background, good lighting
  3. Mark close/nom/far distances on the floor
  4. Wear a fitted top

Per segment:
  1. python3 scripts/ms_67_record_session.py \
       --subject YOU --class POSTURE --distance DIST
  2. Get into pose, wait for 3+ GREEN bars (PASS)
  3. SPACE to start, hold 30 sec, S to save
  4. python3 scripts/ms_63_validate_recording.py \
       --subject YOU --class POSTURE --distance DIST
  5. PASS → next segment; FAIL → reset and re-record

Order:
  All 5 postures at close → all 5 at nom → all 5 at far
  Total: 15 segments, ~20 min
```

---

**Questions?** Ask the researcher (you) before you start recording. It's much
faster to clarify a posture verbally than to re-record after the fact.
