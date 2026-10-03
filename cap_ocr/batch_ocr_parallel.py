#!/usr/bin/env python3
"""Multi-process parallel OCR — one OCR instance per worker."""
import os, sys, glob, time, re
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image

# ---- CONFIG ----
FRAMES_DIR  = os.path.expanduser(os.environ.get("FRAMES_DIR",  "~/cap_ocr/frames"))
OUTPUT_FILE = os.path.expanduser(os.environ.get("OUTPUT_FILE", "~/cap_ocr/data.txt"))
START_FRAME = int(os.environ.get("START_FRAME", "1"))
END_FRAME   = int(os.environ.get("END_FRAME",   "99999"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

DMM_W, DMM_H     = 148, 78
TIMER_W, TIMER_H = 141, 61
SEARCH_X, SEARCH_Y = 500, 100
SEARCH_W, SEARCH_H = 400, 250
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ---- GLOBAL PER-WORKER STATE ----
_ocr = None
_template = None


def init_worker():
    """Called once per worker process — load models + template."""
    global _ocr, _template
    sys.path.insert(0, os.path.expanduser("~/ocr_env/lib/python3.12/site-packages"))
    from pyturboocr import OCR
    _ocr = OCR(tier="tiny", providers=PROVIDERS)
    tpl_frame = Image.open(os.path.join(FRAMES_DIR, "frame_1000.png")).convert("L")
    _template = np.array(tpl_frame.crop((603, 181, 603 + DMM_W, 181 + DMM_H)))


def preprocess_dmm(pil_rgb):
    r, _, _ = pil_rgb.convert("RGB").split()
    arr = np.array(r).astype(np.float32)
    lo, hi = 0.30 * 255, 0.70 * 255
    arr = (arr - lo) / (hi - lo) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((img.width * 6, img.height * 6), Image.LANCZOS)


def preprocess_timer(pil_rgb):
    return pil_rgb.resize((pil_rgb.width * 4, pil_rgb.height * 4), Image.LANCZOS)


def find_dmm_position(frame_gray):
    search = frame_gray[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W]
    if search.shape[0] < _template.shape[0] or search.shape[1] < _template.shape[1]:
        return None
    res = cv2.matchTemplate(search, _template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.5:
        return None
    return (SEARCH_X + max_loc[0], SEARCH_Y + max_loc[1])


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


def process_one(src):
    """Process a single frame — runs in a worker process."""
    num = int(os.path.basename(src).replace("frame_", "").replace(".png", ""))
    t_vid = num / 60.0
    try:
        full = Image.open(src)
        gray = np.array(full.convert("L"))
        pos = find_dmm_position(gray)
        if pos is None:
            return (num, t_vid, None, None, "track_failed")
        x, y = pos
        dmm_pil   = full.crop((x, y, x + DMM_W, y + DMM_H))
        timer_pil = full.crop((x + 111, y + 61, x + 111 + TIMER_W, y + 61 + TIMER_H))

        dmm_img = preprocess_dmm(dmm_pil)
        timer_img = preprocess_timer(timer_pil)
        dmm_img.save(f"/tmp/par_dmm_{num}.png")
        timer_img.save(f"/tmp/par_timer_{num}.png")

        try:
            dmm_text = extract_dmm(_ocr.recognize_image(f"/tmp/par_dmm_{num}.png").text)
        except Exception as e:
            dmm_text = ""
        try:
            timer_text = extract_timer(_ocr.recognize_image(f"/tmp/par_timer_{num}.png").text)
        except Exception as e:
            timer_text = ""

        os.unlink(f"/tmp/par_dmm_{num}.png")
        os.unlink(f"/tmp/par_timer_{num}.png")

        return (num, t_vid, dmm_text, timer_text, None)
    except Exception as e:
        return (num, t_vid, None, None, str(e))


def main():
    all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
    frames = [f for f in all_frames
              if START_FRAME <= int(os.path.basename(f).replace("frame_","").replace(".png","")) <= END_FRAME]
    total = len(frames)
    print(f"# Processing {total} frames with {NUM_WORKERS} workers", file=sys.stderr)

    out = open(OUTPUT_FILE, "w")
    out.write(f"# Parallel OCR\n# Range: {START_FRAME}..{END_FRAME}\n")
    out.write(f"# Format: Volts: {{DMM}} Time: {{TIMER}} ({{SECONDS}})  (frame N, t=X.XXXs)\n\n")

    t0 = time.time()
    good = bad = 0
    processed = 0

    with Pool(NUM_WORKERS, initializer=init_worker) as pool:
        for num, t_vid, dmm_text, timer_text, err in pool.imap_unordered(process_one, frames, chunksize=8):
            if dmm_text is None:
                out.write(f"Volts: ? Time: ? (?)  (frame {num}, t={t_vid:.3f}s, {err})\n")
                bad += 1
            else:
                secs = timer_to_seconds(timer_text or "")
                secs_str = f"{secs:.2f}" if secs is not None else "?"
                out.write(f"Volts: {dmm_text or '?'} Time: {timer_text or '?'} ({secs_str})  "
                          f"(frame {num}, t={t_vid:.3f}s)\n")
                if dmm_text and timer_text: good += 1
                else: bad += 1
            processed += 1
            if processed % 50 == 0 or processed == total:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"[{processed}/{total}] good={good} bad={bad} "
                      f"rate={rate:.1f}f/s ETA={eta:.0f}s", file=sys.stderr)
                out.flush()

    out.close()
    print(f"\n# DONE: {good} good, {bad} bad, total {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
