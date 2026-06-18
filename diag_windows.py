"""Throwaway diagnostic: load a click-UI prescan pickle and run the REAL
window machine to see what speaking windows actually exist for the target,
their durations, and the guardrail-1 overlap fractions. Read-only."""
import pickle, sys
from collections import Counter

from layer1_enrollment.params import EnrollmentParams
from layer1_enrollment.window_machine import WindowMachine, FrameObs
from layer1_enrollment.geometry import yaw_suspends_mar
from layer1_enrollment.vision import FrameFaces, FaceObs  # noqa: needed for unpickle

PKL = sys.argv[1]
P = EnrollmentParams()


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # embeddings are L2-normalized


def vad_near(pts, segments, tol):
    return any(s - tol <= pts <= e + tol for s, e in segments)


with open(PKL, "rb") as f:
    blob = pickle.load(f)

frames = blob["frames"]
fa = blob["file_audio"]
segs = fa.silero_segments_local_ms
print(f"frames={len(frames)}  dur={fa.duration_ms}ms  "
      f"audio_start_pts={fa.audio_start_pts_ms}  vad_segments={len(segs)}")

# Face-count distribution per frame.
dist = Counter(len(fr.faces) for fr in frames)
print("faces-per-frame:", dict(sorted(dist.items())))

# Greedy identity clustering across all faces (cosine >= reid threshold).
clusters = []  # list of [centroid_embedding, count, [det_scores]]
for fr in frames:
    for face in fr.faces:
        placed = False
        for c in clusters:
            if cosine(face.embedding, c[0]) >= P.face_reid_threshold:
                c[1] += 1
                c[2].append(face.det_score)
                placed = True
                break
        if not placed:
            clusters.append([list(face.embedding), 1, [face.det_score]])
clusters.sort(key=lambda c: -c[1])
print(f"\n{len(clusters)} identity cluster(s) (cosine>={P.face_reid_threshold}):")
for i, c in enumerate(clusters[:6]):
    avg = sum(c[2]) / len(c[2])
    print(f"  id{i}: appears in {c[1]} face-instances, avg det={avg:.2f}")

# Treat the chosen cluster as the TARGET (default 0), the other big one as interviewer.
TGT_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OTH_IDX = 1 if TGT_IDX == 0 else 0
print(f"\n*** running with id{TGT_IDX} as TARGET ***")
target = clusters[TGT_IDX][0]
interviewer_emb = clusters[OTH_IDX][0] if len(clusters) > 1 else None


def match(faces, emb):
    best, best_sim = None, -1.0
    for f in faces:
        s = cosine(f.embedding, emb)
        if s >= P.face_reid_threshold and s > best_sim:
            best, best_sim = f, s
    return best


# Build obs the same way enrollment._build_obs does, then run the machine.
obs = []
for fr in frames:
    tgt = match(fr.faces, target)
    others = [f for f in fr.faces if f is not tgt]
    if interviewer_emb is not None:
        intv = match(others, interviewer_emb)
    else:
        intv = max(others, key=lambda f: f.det_score) if others else None
    obs.append(FrameObs(
        pts_ms=fr.pts_ms,
        target_present=tgt is not None,
        target_mar=tgt.mar if tgt else None,
        target_suspended=bool(tgt and yaw_suspends_mar(tgt.yaw_degrees, P)),
        interviewer_present=intv is not None,
        interviewer_mar=intv.mar if intv else None,
        vad_speech=vad_near(fr.pts_ms, segs, P.vad_tol_ms),
    ))

m = WindowMachine(P)
wins = []
for o in obs:
    e = m.step(o)
    if e:
        wins.append(e)
if obs:
    tail = m.finalize(obs[-1].pts_ms)
    if tail:
        wins.append(tail)

print(f"\n{len(wins)} candidate window(s) for the largest identity:")
print(f"  seed_min_ms={P.seed_min_ms}  overlap_max={P.click_overlap_max_frac}  "
      f"yaw_max={P.yaw_max_degrees}  mar_on={P.mar_on}")
clickable = 0
for w in wins:
    open_frac = None
    if w.interviewer_present_frames > 0:
        open_frac = 1.0 - w.interviewer_closed_frames / w.interviewer_present_frames
    long_enough = w.duration_ms >= P.seed_min_ms
    overlap_ok = (open_frac is None) or (open_frac <= P.click_overlap_max_frac)
    ok = long_enough and overlap_ok
    clickable += ok
    of = "n/a" if open_frac is None else f"{open_frac:.0%}"
    print(f"  [{w.t_start_ms:>6}..{w.t_stop_ms:>6}] "
          f"dur={w.duration_ms:>5}ms reason={w.end_reason:<22} "
          f"int_open={of:>4} -> {'CLICKABLE' if ok else 'blocked'}"
          f"{'' if long_enough else ' (too short)'}"
          f"{'' if overlap_ok else ' (overlap)'}")
print(f"\n=> {clickable} clickable window(s).")

# Where are the single-face stretches (guardrail 1 cannot false-trigger)?
solo = [o.pts_ms for o, fr in zip(obs, frames)
        if len(fr.faces) == 1 and o.vad_speech and not o.target_suspended]
print(f"\nsingle-face + VAD + not-suspended frames: {len(solo)}")
if solo:
    runs, start, prev = [], solo[0], solo[0]
    step = (frames[1].pts_ms - frames[0].pts_ms) if len(frames) > 1 else 40
    for t in solo[1:]:
        if t - prev > step * 3:
            runs.append((start, prev)); start = t
        prev = t
    runs.append((start, prev))
    runs.sort(key=lambda r: -(r[1] - r[0]))
    print("longest solo+speech stretches (ms):")
    for s, e in runs[:8]:
        print(f"  {s:>6}..{e:>6}  ({e - s}ms)")
