#!/usr/bin/env python3
import os, sys, glob, time, re
import numpy as np
from PIL import Image
import cv2
from pyturboocr import OCR

FRAMES_DIR  = os.path.expanduser(os.environ.get("FRAMES_DIR",  "~/cap_ocr/frames"))
OUTPUT_FILE = os.path.expanduser(os.environ.get("OUTPUT_FILE", "~/cap_ocr/data.txt"))
START_FRAME = int(os.environ.get("START_FRAME", "1"))
END_FRAME   = int(os.environ.get("END_FRAME",   "99999"))

DMM_W, DMM_H     = 148, 78
TIMER_W, TIMER_H = 141, 61

SEARCH_X, SEARCH_Y = 500, 100
SEARCH_W, SEARCH_H = 400, 250

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
print(f"[info] providers: {PROVIDERS}", file=sys.stderr)
ocr = OCR(tier="tiny", providers=PROVIDERS)


def preprocess_dmm(img_pil):
    r, g, b = img_pil.convert("RGB").split()
    arr = np.array(r).astype(np.float32)
    # Fixed levels matching ImageMagick -level 30%,70%
    lo, hi = 0.30 * 255, 0.70 * 255
    arr = (arr - lo) / (hi - lo) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((img.width * 6, img.height * 6), Image.LANCZOS)

def preprocess_timer(img_pil):
    return img_pil.resize((img_pil.width * 4, img_pil.height * 4), Image.LANCZOS)


def find_dmm_position(frame_gray):
    search = frame_gray[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W]
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


def timer_to_seconds(s):
    m = re.match(r"^(\d+):(\d{2})\.?(\d*)$", s)
    if not m: return None
    mins, secs, frac = int(m.group(1)), int(m.group(2)), m.group(3) or "0"
    return mins * 60 + secs + (float("0." + frac) if frac else 0.0)


# Build template once
tpl_frame = Image.open(os.path.join(FRAMES_DIR, "frame_1000.png")).convert("L")
template = np.array(tpl_frame.crop((603, 181, 603 + DMM_W, 181 + DMM_H)))
print(f"[info] template shape: {template.shape}", file=sys.stderr)


all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
frames = [f for f in all_frames
          if START_FRAME <= int(os.path.basename(f).replace("frame_","").replace(".png","")) <= END_FRAME]
total = len(frames)
print(f"# Processing {total} frames ({START_FRAME}..{END_FRAME})", file=sys.stderr)

out = open(OUTPUT_FILE, "w")
out.write(f"# Batch OCR results\n# Range: {START_FRAME}..{END_FRAME}\n")
out.write(f"# Format: Volts: {{DMM}} Time: {{TIMER}} ({{SECONDS}})  (frame N, t=X.XXXs)\n\n")

t0 = time.time()
good, bad = 0, 0

for i, src in enumerate(frames, 1):
    num  = int(os.path.basename(src).replace("frame_","").replace(".png",""))
    t_vid = num / 60.0

    full = Image.open(src)
    gray = np.array(full.convert("L"))
    pos, score = find_dmm_position(gray)

    if pos is None:
        out.write(f"Volts: ? Time: ? (?)  (frame {num}, t={t_vid:.3f}s, TRACK FAILED)\n")
        bad += 1
    else:
        x, y = pos
        dmm_crop   = full.crop((x, y, x + DMM_W, y + DMM_H))
        timer_crop = full.crop((x + 111, y + 61, x + 111 + TIMER_W, y + 61 + TIMER_H))

        try:
            dmm_img = preprocess_dmm(dmm_crop)
            dmm_img.save("/tmp/dmm_ocr.png")
            dmm_text = extract_dmm(ocr.recognize_image("/tmp/dmm_ocr.png").text)
        except Exception as e:
            dmm_text = ""
            print(f"[warn] frame {num} dmm: {e}", file=sys.stderr)

        try:
            timer_img = preprocess_timer(timer_crop)
            timer_img.save("/tmp/timer_ocr.png")
            timer_text = extract_timer(ocr.recognize_image("/tmp/timer_ocr.png").text)
        except Exception as e:
            timer_text = ""
            print(f"[warn] frame {num} timer: {e}", file=sys.stderr)

        secs = timer_to_seconds(timer_text)
        secs_str = f"{secs:.2f}" if secs is not None else "?"

        out.write(f"Volts: {dmm_text or '?'} Time: {timer_text or '?'} ({secs_str})  "
                  f"(frame {num}, t={t_vid:.3f}s)\n")

        if dmm_text and timer_text: good += 1
        else: bad += 1

    if i % 25 == 0 or i == total:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        print(f"[{i}/{total}] good={good} bad={bad} rate={rate:.2f}f/s ETA={eta/60:.1f}min",
              file=sys.stderr)
        out.flush()

out.close()
print(f"\n# DONE: {good} good, {bad} bad, total {total}", file=sys.stderr)
