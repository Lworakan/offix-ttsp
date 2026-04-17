# Forward-Head Recording Guide

This guide is for re-collecting `forward_head` data, especially for any
subject whose original recording failed validation (e.g. **pan**: 1/11 metrics
clear). Following this protocol prevents the failure mode where the recorded
pose is visually similar to a normal sit and not detectable by any quantitative
metric.

---

## What "forward_head" means in this dataset

A clinically meaningful forward-head posture means the **head moves forward of
the line through the shoulders**, with one or more of the following effects:

| Cue | Description |
|---|---|
| **Translation** | The whole head shifts forward in z (toward the camera) by ≥ 3 cm |
| **Cervical extension** | Chin juts forward of the ears (the most common natural form) |
| **Cervical flexion** | Head tips down while still being forward of the shoulders |

**Any of these is acceptable.** Different subjects use different combinations.
The important thing is that *quantitative metrics* register the deviation.

---

## What goes wrong (the pan failure mode)

The pan recording failed because the head moved **less than 5 mm** forward of
the shoulders across all frames. From the camera's perspective the pose is
visually almost identical to neutral sit. None of 11 independent metrics
(world coords, image coords, DA3 depth, hip-relative, etc.) showed clear
separation. The model has nothing to learn from such a recording.

To avoid this:

1. **Push the pose**. A subtle head shift is invisible to monocular sensors.
   Push your head forward until you feel mild stretch in the back of the neck.
2. **Hold consistently**. Mid-recording drifting back into neutral is bad.
3. **Verify with the live tool** before keeping the recording.

---

## Step-by-step recording protocol

### Setup (once per subject, before any recording)

1. Sit on a stable chair, feet flat on floor.
2. Laptop on a desk, webcam at chin level (neutral sit head height).
3. Plain background, decent lighting (no strong backlight).
4. Make sure your `sit_straight` recording is already done **at the same camera
   position**. The validator uses your sit_straight as the baseline. If you
   have not recorded sit_straight yet, do that first.

### For each distance (close, nom, far)

| Distance | Camera-to-chest | How to verify |
|---|---|---|
| **close** | ~50 cm | One forearm length from desk edge to your chest |
| **nom**   | ~70 cm | Comfortable laptop typing position |
| **far**   | ~90 cm | Sitting back, arms slightly extended |

Mark the chair position with tape so you can return to it.

### Record one forward_head segment

1. Open a terminal in the project directory.
2. Run:
   ```bash
   python3 scripts/ms_67_record_fh_session.py --subject pan --distance close
   ```
   Replace `pan` with your subject id and `close` with the distance.
3. The webcam window opens. You will see your skeleton overlay and a panel on
   the right showing **11 live metrics** with z-score bars.
4. **Slowly push your head forward** while watching the panel:
   - Bars turn **green** when the metric reaches |z| ≥ 1.0 (clear separation)
   - You need **at least 3 green bars at the same time** to enter PASS state
   - The status bar at the top of the screen says **PASS** when ready
5. Press **SPACE** to start recording. The buffer will only fill while you are
   in PASS state — if you drift back, recording silently pauses.
6. Hold the pose. Keep watching the bars. Try to push for 4–5 green bars,
   not just the bare minimum 3.
7. When the buffer reaches the target (`saved=180 frames`), the screen shows
   "TARGET REACHED — press S to save".
8. Press **S**. Frames and landmarks are saved.

### Validate immediately

After saving, run the offline validator:
```bash
python3 scripts/ms_63_validate_recording.py --subject pan --class forward_head --distance close
```

Expected output:
```
[pan] forward_head/close  n_frames=180  n_clear=N/11  verdict=PASS
```
- **PASS** with `n_clear ≥ 3` → keep the recording, move on
- **FAIL** → press R to reset and re-record more strongly

### Repeat for the other two distances

Don't move on to the next distance until the current one passes validation.

---

## Common mistakes and how to fix them

| Mistake | Symptom | Fix |
|---|---|---|
| Head shifted only 1–2 cm | All bars stay grey or yellow | Push harder. Imagine reading small text 5 cm closer. |
| Posture drifts mid-recording | Some frames PASS, some don't | Brace your back against the chair so only your head moves. |
| Looking down (head tilt) instead of forward | `nose_y-sh_y` and `cva` clear, but `nose_z-sh_z` flat | Keep eye level on the screen — translate, don't tilt. |
| Body leans forward at the hips | Hip-relative metrics strong but head-relative weak | Your hips should NOT move. Only your neck. |
| Recording starts before reaching PASS | `saved=0` even though you're posing | Wait for the green PASS state before pressing SPACE. |
| Camera too far / head too small in frame | MediaPipe loses track, "NO POSE" | Move closer or raise camera. |

---

## When to give up and move on

If after **5 attempts** you cannot reach PASS for a particular distance:

1. Check that your sit_straight baseline is correct (run validator on your sit_straight first; if its own metrics are weird, re-record sit_straight).
2. Try the next distance (some subjects fail at one distance but pass at others — that's OK, partial coverage is still useful).
3. If still failing, ask another person to evaluate the pose visually. Sometimes you think you're posing forward but the camera disagrees.

---

## Reference: the 11 metrics

| Metric | Captures |
|---|---|
| `nose_z-sh_z` | head position relative to shoulders, depth axis |
| `nose_y-sh_y` | head height drop |
| `chin_jut`    | nose forward of ears (cervical extension) |
| `cva`         | ear-shoulder line angle (smaller = more forward) |
| `img_h`       | horizontal head offset, image space |
| `img_v`       | vertical head offset, image space |
| `nose_z-hip_z`| whole-torso forward lean |
| `sh_z-hip_z`  | shoulder-hip lean (whole upper body) |
| `nose_y-hip_y`| head height vs hip |
| `sh-hip_x`    | left/right tilt sanity |
| `torso_curl`  | trunk compression |

A valid forward_head recording will show **at least 3 of these as |z| ≥ 1.0**
relative to the same subject's own sit_straight baseline. The cleanest
recordings (peemai, nonny in the original dataset) reach 7–10 clear metrics.

---

## Quick reference card

```
1. Record sit_straight (or check it's already done)
2. python3 scripts/ms_67_record_fh_session.py --subject X --distance close
3. Push head forward until ≥3 green bars (PASS state)
4. SPACE to start, hold 30 sec, S to save
5. python3 scripts/ms_63_validate_recording.py --subject X --class forward_head --distance close
6. PASS → next distance; FAIL → reset and re-record
7. Repeat for nom, far
```

When all 3 distances pass, run a full subject audit:
```bash
python3 scripts/ms_63_validate_recording.py --subject X
```
