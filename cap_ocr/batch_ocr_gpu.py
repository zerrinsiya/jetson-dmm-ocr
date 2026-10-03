#!/usr/bin/env python3
"""
Batch OCR all frames → data.txt
Handles camera drift via template matching, and won't crash on bad frames.

Usage:
  Local test:
    FRAMES_DIR=~/Downloads/frames \
    OUTPUT_FILE=~/Downloads/data_local_test.txt \
    START_FRAME=1000 END_FRAME=1200 \
    python3 batch_ocr.py

  Remote:
    python3 batch_ocr.py    # uses ~/cap_ocr/frames + ~/cap_ocr/data.txt
"""

import os, sys, glob, subprocess, re, time
import cv2
from pyturboocr import OCR

# ---- CONFIG (env-var overridable) ----
FRAMES_DIR  = os.path.expanduser(os.environ.get("FRAMES_DIR",  "~/cap_ocr/frames"))
OUTPUT_FILE = os.path.expanduser(os.environ.get("OUTPUT_FILE", "~/cap_ocr/data.txt"))
START_FRAME = int(os.environ.get("START_FRAME", "1"))
END_FRAME   = int(os.environ.get("END_FRAME",   "99999"))

DMM_W, DMM_H     = 148, 78
TIMER_W, TIMER_H = 141, 61
DMM_PRE   = ["-channel","R","-separate","+channel",
             "-normalize","-level","30%,70%","-resize","600%"]
TIMER_PRE = ["-resize","400%"]

SEARCH_X, SEARCH_Y = 500, 100
SEARCH_W, SEARCH_H = 400, 250

# ---- ImageMagick binary: use `magick` if present, else `convert` ----
import shutil
CONVERT = "magick" if shutil.which("magick") else "convert"
print(f"[info] using {CONVERT}", file=sys.stderr)

# Template source — the DMM at frame 1000
TEMPLATE_FRAME = os.path.join(FRAMES_DIR, "frame_1000.png")
TEMPLATE_CROP  = f"{DMM_W}x{DMM_H}+603+181"

# ---- OCR backend ----

import os
os.makedirs("/tmp/cuda_cache", exist_ok=True)

PROVIDERS = [
    ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
    }),
    "CPUExecutionProvider",
]
print(f"[info] providers: {PROVIDERS}", file=sys.stderr)

ocr = OCR(tier="tiny", providers=PROVIDERS)

# ---- Build template ----
print(f"[info] template from {TEMPLATE_FRAME}", file=sys.stderr)
subprocess.run(
    [CONVERT, TEMPLATE_FRAME, "+repage", "-crop", TEMPLATE_CROP,
     "+repage", "/tmp/dmm_template.png"],
    check=True
)
template = cv2.imread("/tmp/dmm_template.png", cv2.IMREAD_GRAYSCALE)
if template is None:
    raise SystemExit("Failed to build DMM template")
print(f"[info] template shape: {template.shape}", file=sys.stderr)


def find_dmm_position(frame_path):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0.0
    search = img[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.5:
        return None, max_val
    return (SEARCH_X + max_loc[0], SEARCH_Y + max_loc[1]), max_val


def extract_dmm(text):
    first = text.split("\n")[0].strip()
    m = re.match(r"^(-?\d+\.?\d*)", first)
    if m: return m.group(1)
    m = re.search(r"-?\d+\.\d+", text)
    return m.group(0) if m else ""


def extract_timer(text):
    m = re.search(r"\d+:\d{2}\.?\d*", text)
    return m.group(0) if m else ""


def timer_to_seconds(timer_str):
    m = re.match(r"^(\d+):(\d{2})\.?(\d*)$", timer_str)
    if not m: return None
    mins = int(m.group(1)); secs = int(m.group(2))
    frac = m.group(3) or "0"
    return mins * 60 + secs + (float("0." + frac) if frac else 0.0)


def safe_crop_and_ocr(src, crop_spec, pre_ops, ocr_fn):
    try:
        subprocess.run(
            [CONVERT, src, "+repage", "-crop", crop_spec, "+repage",
             *pre_ops, "/tmp/_ocr_tmp.png"],
            check=True, capture_output=True
        )
    except Exception as e:
        return "", f"convert: {e}"
    if not os.path.exists("/tmp/_ocr_tmp.png"):
        return "", "convert produced no file"
    try:
        return ocr_fn("/tmp/_ocr_tmp.png").text, None
    except Exception as e:
        return "", f"ocr: {e}"


# ---- MAIN ----
all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
frames = [f for f in all_frames
          if START_FRAME <= int(os.path.basename(f).replace("frame_","").replace(".png","")) <= END_FRAME]
total = len(frames)
print(f"# Processing {total} frames ({START_FRAME}..{END_FRAME})", file=sys.stderr)

out = open(OUTPUT_FILE, "w")
out.write(f"# Batch OCR results\n")
out.write(f"# Range: frame {START_FRAME} to {END_FRAME}\n")
out.write(f"# Format: Volts: {{DMM}} Time: {{TIMER}} ({{SECONDS}})  (frame N, t=X.XXXs)\n\n")

t0 = time.time()
good, bad = 0, 0

for i, src in enumerate(frames, 1):
    base = os.path.basename(src)
    num  = int(base.replace("frame_","").replace(".png",""))
    t_vid = num / 60.0

    pos, score = find_dmm_position(src)
    if pos is None:
        out.write(f"Volts: ? Time: ? (?)  (frame {num}, t={t_vid:.3f}s, TRACK FAILED)\n")
        bad += 1
    else:
        x, y = pos
        dmm_crop   = f"{DMM_W}x{DMM_H}+{x}+{y}"
        timer_crop = f"{TIMER_W}x{TIMER_H}+{x+111}+{y+61}"

        dmm_raw,   dmm_err   = safe_crop_and_ocr(src, dmm_crop,   DMM_PRE,   ocr.recognize_image)
        timer_raw, timer_err = safe_crop_and_ocr(src, timer_crop, TIMER_PRE, ocr.recognize_image)

        dmm_text   = extract_dmm(dmm_raw)     if not dmm_err   else ""
        timer_text = extract_timer(timer_raw) if not timer_err else ""

        secs = timer_to_seconds(timer_text)
        secs_str = f"{secs:.2f}" if secs is not None else "?"

        out.write(f"Volts: {dmm_text or '?'} Time: {timer_text or '?'} ({secs_str})  "
                  f"(frame {num}, t={t_vid:.3f}s)\n")

        if dmm_text and timer_text:
            good += 1
        else:
            bad += 1
            if dmm_err or timer_err:
                print(f"[warn] frame {num}: dmm_err={dmm_err} timer_err={timer_err}",
                      file=sys.stderr)

    if i % 25 == 0 or i == total:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        print(f"[{i}/{total}] good={good} bad={bad} "
              f"rate={rate:.2f}f/s ETA={eta/60:.1f}min",
              file=sys.stderr)
        out.flush()

out.close()
print(f"\n# DONE: {good} good, {bad} bad, total {total}", file=sys.stderr)
