#!/usr/bin/env python3
"""
Fully batched OCR: detection + recognition both batched.
Keeps only boxes whose recognized text matches DMM/timer patterns.
"""
import os, sys, glob, time, re
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, os.path.expanduser("~/ocr_env/lib/python3.12/site-packages"))
from pyturboocr.preprocess import (
    resize_for_detection, normalize_chw, crop_and_rectify, resize_for_recognition,
)
from pyturboocr.postprocess.db import DBParams, boxes_from_probability_map
from pyturboocr.postprocess.ctc import load_charset, ctc_greedy_decode

# ---- CONFIG ----
FRAMES_DIR  = os.path.expanduser(os.environ.get("FRAMES_DIR",  "~/cap_ocr/frames"))
OUTPUT_FILE = os.path.expanduser(os.environ.get("OUTPUT_FILE", "~/cap_ocr/data.txt"))
START_FRAME = int(os.environ.get("START_FRAME", "1"))
END_FRAME   = int(os.environ.get("END_FRAME",   "99999"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "32"))

DMM_W, DMM_H     = 148, 78
TIMER_W, TIMER_H = 141, 61
SEARCH_X, SEARCH_Y = 500, 100
SEARCH_W, SEARCH_H = 400, 250

MODEL_DIR = os.path.expanduser("~/.cache/pyturboocr")
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ---- MODELS ----
print("[info] loading models", file=sys.stderr)
det_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "det_tiny.onnx"), providers=PROVIDERS)
rec_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "rec_tiny.onnx"), providers=PROVIDERS)
det_in = det_sess.get_inputs()[0].name
rec_in = rec_sess.get_inputs()[0].name
charset = load_charset(os.path.join(MODEL_DIR, "keys_tiny.txt"))
print(f"[info] det providers: {det_sess.get_providers()}", file=sys.stderr)
print(f"[info] charset: {len(charset)}", file=sys.stderr)

# ---- TEMPLATE ----
tpl_frame = Image.open(os.path.join(FRAMES_DIR, "frame_1000.png")).convert("L")
template = np.array(tpl_frame.crop((603, 181, 603 + DMM_W, 181 + DMM_H)))


def find_dmm_position(frame_gray):
    search = frame_gray[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.5:
        return None
    return (SEARCH_X + max_loc[0], SEARCH_Y + max_loc[1])


def preprocess_dmm_bgr(pil_rgb):
    r, _, _ = pil_rgb.split()
    arr = np.array(r).astype(np.float32)
    lo, hi = 0.30 * 255, 0.70 * 255
    arr = (arr - lo) / (hi - lo) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.resize(bgr, (bgr.shape[1]*6, bgr.shape[0]*6), interpolation=cv2.INTER_LANCZOS4)


def preprocess_timer_bgr(pil_rgb):
    bgr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (bgr.shape[1]*4, bgr.shape[0]*4), interpolation=cv2.INTER_LANCZOS4)


def run_detection_batch(crops_bgr):
    """
    Run detection on a batch of BGR images.
    Because input sizes vary, we resize each to a common padded size.
    Returns: list of (boxes_orig_coords, crop_bgr) — one entry per input crop.
    """
    # Resize each crop and record scaling
    prepared = []  # (resized, ratio_h, ratio_w)
    max_h = max_w = 0
    for c in crops_bgr:
        resized, rh, rw = resize_for_detection(c)
        prepared.append((resized, rh, rw))
        max_h = max(max_h, resized.shape[0])
        max_w = max(max_w, resized.shape[1])
    # Pad to common size
    n = len(prepared)
    tensor = np.zeros((n, 3, max_h, max_w), dtype=np.float32)
    for i, (resized, _, _) in enumerate(prepared):
        h, w = resized.shape[:2]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (rgb - mean) / std
        chw = normalized.transpose(2, 0, 1)
        tensor[i, :, :h, :w] = chw
    (raw_out,) = det_sess.run(None, {det_in: tensor})
    # raw_out shape: (n, 1, max_h, max_w)
    results = []
    for i, (resized, rh, rw) in enumerate(prepared):
        h, w = resized.shape[:2]
        prob_map = raw_out[i, 0, :h, :w]
        if prob_map.min() < 0 or prob_map.max() > 1:
            prob_map = 1.0 / (1.0 + np.exp(-prob_map))
        boxes_scored = boxes_from_probability_map(prob_map, DBParams())
        # Map boxes back to original crop coordinates
        scale_h = prob_map.shape[0] / resized.shape[0]
        scale_w = prob_map.shape[1] / resized.shape[1]
        mapped = []
        for box, _score in boxes_scored:
            b = box.copy()
            b[:, 0] = b[:, 0] / scale_w / rw
            b[:, 1] = b[:, 1] / scale_h / rh
            mapped.append(b)
        results.append(mapped)
    return results


def run_recognition_batch(crops_bgr):
    """Batched recognition. crops_bgr: list of BGR crops (variable size)."""
    if not crops_bgr:
        return []
    n = len(crops_bgr)
    REC_H, REC_W = 48, 320
    tensor = np.zeros((n, 3, REC_H, REC_W), dtype=np.float32)
    for i, crop in enumerate(crops_bgr):
        h, w = crop.shape[:2]
        ratio = w / float(h)
        rw = min(int(np.ceil(REC_H * ratio)), REC_W)
        rw = max(rw, 1)
        resized = cv2.resize(crop, (rw, REC_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5
        tensor[i, :, :, :rw] = rgb.transpose(2, 0, 1)
    (logits,) = rec_sess.run(None, {rec_in: tensor})
    return ctc_greedy_decode(logits, charset)


def pick_dmm_reading(texts_and_boxes):
    """Pick the box whose text looks like a DMM voltage reading."""
    best = ("", 0.0)
    for text, conf in texts_and_boxes:
        m = re.match(r"^(-?\d+\.\d+)", text.strip())
        if m:
            return (m.group(1), conf)
    return best


def pick_timer_reading(texts_and_boxes):
    """Pick the box whose text looks like a timer M:SS.cc."""
    for text, conf in texts_and_boxes:
        m = re.search(r"(\d+:\d{2}\.?\d*)", text)
        if m:
            return (m.group(1), conf)
    return ("", 0.0)


def timer_to_seconds(s):
    m = re.match(r"^(\d+):(\d{2})\.?(\d*)$", s)
    if not m: return None
    mins, secs, frac = int(m.group(1)), int(m.group(2)), m.group(3) or "0"
    return mins * 60 + secs + (float("0." + frac) if frac else 0.0)


# ---- MAIN ----
all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
frames = [f for f in all_frames
          if START_FRAME <= int(os.path.basename(f).replace("frame_","").replace(".png","")) <= END_FRAME]
total = len(frames)
print(f"# Processing {total} frames, batch={BATCH_SIZE}", file=sys.stderr)

out = open(OUTPUT_FILE, "w")
out.write(f"# Batched OCR v2 (det+rec)\n# Range: {START_FRAME}..{END_FRAME}\n")
out.write(f"# Format: Volts: {{DMM}} Time: {{TIMER}} ({{SECONDS}})  (frame N, t=X.XXXs)\n\n")

t0 = time.time()
good, bad = 0, 0
processed = 0

for batch_start in range(0, total, BATCH_SIZE):
    batch_frames = frames[batch_start:batch_start + BATCH_SIZE]

    # Stage 1: prepare crops
    dmm_crops, timer_crops = [], []
    metas = []
    for src in batch_frames:
        num = int(os.path.basename(src).replace("frame_","").replace(".png",""))
        t_vid = num / 60.0
        try:
            full = Image.open(src)
            gray = np.array(full.convert("L"))
            pos = find_dmm_position(gray)
            if pos is None:
                metas.append({"num": num, "t_vid": t_vid, "ok": False})
                continue
            x, y = pos
            dmm_pil   = full.crop((x, y, x + DMM_W, y + DMM_H))
            timer_pil = full.crop((x + 111, y + 61, x + 111 + TIMER_W, y + 61 + TIMER_H))
            dmm_crops.append(preprocess_dmm_bgr(dmm_pil))
            timer_crops.append(preprocess_timer_bgr(timer_pil))
            metas.append({"num": num, "t_vid": t_vid, "ok": True, "idx": len(dmm_crops) - 1})
        except Exception as e:
            metas.append({"num": num, "t_vid": t_vid, "ok": False, "err": str(e)})

    # Stage 2: batched detection
    dmm_boxes_per_frame   = run_detection_batch(dmm_crops) if dmm_crops else []
    timer_boxes_per_frame = run_detection_batch(timer_crops) if timer_crops else []

    # Stage 3: collect all crops for batched recognition
    dmm_crop_list = []
    dmm_owner = []   # (frame_idx_in_batch, )
    for fi, (boxes, crop_bgr) in enumerate(zip(dmm_boxes_per_frame, dmm_crops)):
        for box in boxes:
            c = crop_and_rectify(crop_bgr, box)
            if c.size:
                dmm_crop_list.append(c)
                dmm_owner.append(fi)

    timer_crop_list = []
    timer_owner = []
    for fi, (boxes, crop_bgr) in enumerate(zip(timer_boxes_per_frame, timer_crops)):
        for box in boxes:
            c = crop_and_rectify(crop_bgr, box)
            if c.size:
                timer_crop_list.append(c)
                timer_owner.append(fi)

    # Stage 4: batched recognition
    dmm_rec = run_recognition_batch(dmm_crop_list)
    timer_rec = run_recognition_batch(timer_crop_list)

    # Stage 5: associate per-frame
    dmm_per_frame = [[] for _ in dmm_crops]
    for (text, conf), fi in zip(dmm_rec, dmm_owner):
        dmm_per_frame[fi].append((text, conf))

    timer_per_frame = [[] for _ in timer_crops]
    for (text, conf), fi in zip(timer_rec, timer_owner):
        timer_per_frame[fi].append((text, conf))

    # Stage 6: write
    for meta in metas:
        num, t_vid = meta["num"], meta["t_vid"]
        if not meta["ok"]:
            out.write(f"Volts: ? Time: ? (?)  (frame {num}, t={t_vid:.3f}s, TRACK FAILED)\n")
            bad += 1
        else:
            fi = meta["idx"]
            dmm_text, _   = pick_dmm_reading(dmm_per_frame[fi])
            timer_text, _ = pick_timer_reading(timer_per_frame[fi])
            secs = timer_to_seconds(timer_text)
            secs_str = f"{secs:.2f}" if secs is not None else "?"
            out.write(f"Volts: {dmm_text or '?'} Time: {timer_text or '?'} ({secs_str})  "
                      f"(frame {num}, t={t_vid:.3f}s)\n")
            if dmm_text and timer_text: good += 1
            else: bad += 1
        processed += 1

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if rate > 0 else 0
    print(f"[{processed}/{total}] good={good} bad={bad} "
          f"rate={rate:.1f}f/s ETA={eta:.0f}s", file=sys.stderr)
    out.flush()

out.close()
print(f"\n# DONE: {good} good, {bad} bad, total {total}", file=sys.stderr)
