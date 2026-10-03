#!/usr/bin/env python3
"""
Batched OCR — no detection, pure recognition on known crop regions.
~10-30x faster than per-image inference.
"""
import os, sys, glob, time, re
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, os.path.expanduser("~/ocr_env/lib/python3.12/site-packages"))
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

REC_H = 48
REC_W = 320

MODEL_DIR = os.path.expanduser("~/.cache/pyturboocr")
REC_MODEL = os.path.join(MODEL_DIR, "rec_tiny.onnx")
KEYS_FILE = os.path.join(MODEL_DIR, "keys_tiny.txt")

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ---- MODEL ----
print(f"[info] loading rec model {REC_MODEL}", file=sys.stderr)
sess = ort.InferenceSession(REC_MODEL, providers=PROVIDERS)
print(f"[info] providers active: {sess.get_providers()}", file=sys.stderr)
input_name = sess.get_inputs()[0].name
charset = load_charset(KEYS_FILE)
print(f"[info] charset size: {len(charset)}", file=sys.stderr)


# ---- TEMPLATE (for DMM tracking) ----
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


def preprocess_dmm_bgr(dmm_rgb):
    """Input: PIL RGB crop. Output: BGR HWC uint8, preprocessed."""
    r, _, _ = dmm_rgb.split()
    arr = np.array(r).astype(np.float32)
    lo, hi = 0.30 * 255, 0.70 * 255
    arr = (arr - lo) / (hi - lo) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # PIL L -> BGR for consistency with rec preprocess (it does cvtColor BGR2RGB)
    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    # Upscale 6x
    bgr = cv2.resize(bgr, (bgr.shape[1] * 6, bgr.shape[0] * 6), interpolation=cv2.INTER_LANCZOS4)
    return bgr


def preprocess_timer_bgr(timer_rgb):
    bgr = cv2.cvtColor(np.array(timer_rgb), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (bgr.shape[1] * 4, bgr.shape[0] * 4), interpolation=cv2.INTER_LANCZOS4)
    return bgr


def rec_preprocess_batch(crops_bgr):
    """Match pyturboocr's resize_for_recognition exactly, but batched.
    Returns (N, 3, 48, 320) float32 tensor."""
    n = len(crops_bgr)
    batch = np.zeros((n, 3, REC_H, REC_W), dtype=np.float32)
    for i, crop in enumerate(crops_bgr):
        h, w = crop.shape[:2]
        ratio = w / float(h)
        resized_w = min(int(np.ceil(REC_H * ratio)), REC_W)
        resized_w = max(resized_w, 1)
        resized = cv2.resize(crop, (resized_w, REC_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5
        chw = rgb.transpose(2, 0, 1)
        batch[i, :, :, :resized_w] = chw
    return batch


def rec_batch(crops_bgr):
    """Run recognition on a batch of crops. Returns list of texts."""
    if not crops_bgr:
        return []
    tensor = rec_preprocess_batch(crops_bgr)
    (logits,) = sess.run(None, {input_name: tensor})
    decoded = ctc_greedy_decode(logits, charset)
    return [t for (t, c) in decoded]


def extract_dmm(text):
    first = text.split("\n")[0].strip()
    m = re.match(r"^(-?\d+\.?\d*)", first)
    if m: return m.group(1)
    m = re.search(r"-?\d+\.\d+", text)
    return m.group(0) if m else ""


def extract_timer(text):
    m = re.search(r"\d+:\d{2}\.?\d*", text)
    return m.group(0) if m else ""


def timer_to_seconds(s):
    m = re.match(r"^(\d+):(\d{2})\.?(\d*)$", s)
    if not m: return None
    mins, secs, frac = int(m.group(1)), int(m.group(2)), m.group(3) or "0"
    return mins * 60 + secs + (float("0." + frac) if frac else 0.0)


# ---- MAIN ----
all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
frames = [f for f in all_frames
          if START_FRAME <= int(os.path.basename(f).replace("frame_", "").replace(".png", "")) <= END_FRAME]
total = len(frames)
print(f"# Processing {total} frames ({START_FRAME}..{END_FRAME}), batch={BATCH_SIZE}", file=sys.stderr)

out = open(OUTPUT_FILE, "w")
out.write(f"# Batched OCR results\n# Range: {START_FRAME}..{END_FRAME}\n")
out.write(f"# Format: Volts: {{DMM}} Time: {{TIMER}} ({{SECONDS}})  (frame N, t=X.XXXs)\n\n")

t0 = time.time()
good, bad = 0, 0
processed = 0

for batch_start in range(0, total, BATCH_SIZE):
    batch_frames = frames[batch_start:batch_start + BATCH_SIZE]

    # ---- Stage 1: prep all frames in batch (CPU) ----
    metas = []
    dmm_crops, timer_crops = [], []
    for src in batch_frames:
        num = int(os.path.basename(src).replace("frame_", "").replace(".png", ""))
        t_vid = num / 60.0
        try:
            full = Image.open(src)
            gray = np.array(full.convert("L"))
            pos = find_dmm_position(gray)
            if pos is None:
                metas.append({"num": num, "t_vid": t_vid, "ok": False})
                continue
            x, y = pos
            dmm_crop   = full.crop((x, y, x + DMM_W, y + DMM_H))
            timer_crop = full.crop((x + 111, y + 61, x + 111 + TIMER_W, y + 61 + TIMER_H))
            dmm_crops.append(preprocess_dmm_bgr(dmm_crop))
            timer_crops.append(preprocess_timer_bgr(timer_crop))
            metas.append({"num": num, "t_vid": t_vid, "ok": True,
                          "idx": len(dmm_crops) - 1})
        except Exception as e:
            metas.append({"num": num, "t_vid": t_vid, "ok": False, "err": str(e)})

    # ---- Stage 2: batched OCR (GPU) ----
    dmm_texts   = rec_batch(dmm_crops)
    timer_texts = rec_batch(timer_crops)

    # ---- Stage 3: write results ----
    for meta in metas:
        num, t_vid = meta["num"], meta["t_vid"]
        if not meta["ok"]:
            out.write(f"Volts: ? Time: ? (?)  (frame {num}, t={t_vid:.3f}s, TRACK FAILED)\n")
            bad += 1
        else:
            i = meta["idx"]
            dmm_text   = extract_dmm(dmm_texts[i])
            timer_text = extract_timer(timer_texts[i])
            secs = timer_to_seconds(timer_text)
            secs_str = f"{secs:.2f}" if secs is not None else "?"
            out.write(f"Volts: {dmm_text or '?'} Time: {timer_text or '?'} ({secs_str})  "
                      f"(frame {num}, t={t_vid:.3f}s)\n")
            if dmm_text and timer_text: good += 1
            else: bad += 1
        processed += 1

    if processed % BATCH_SIZE == 0 or processed == total:
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        print(f"[{processed}/{total}] good={good} bad={bad} "
              f"rate={rate:.1f}f/s ETA={eta:.0f}s", file=sys.stderr)
        out.flush()

out.close()
print(f"\n# DONE: {good} good, {bad} bad, total {total}", file=sys.stderr)
