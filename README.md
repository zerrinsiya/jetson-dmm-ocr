# DMM + Timer OCR from Video

Extract voltage and timestamp readings from a phone-recorded video of a DMM display and a parallel stopwatch, then reconstruct the capacitor discharge curve.

## Hardware on my side

### Target machine (all processing was done here)
- Jetson Orin Nano Super 8GB
- JetPack 7.2 (L4T r39.2)
- CUDA 13.2, TensorRT 10.16, cuDNN 9
- Ubuntu 24.04, Python 3.12
- 1024-core Ampere GPU (sm_87)
- 6-core ARM Cortex-A78AE
- NVMe: 915.1 GB (768.2 GB free during build)

### Local test machine (used during testing and development)
- Fedora Workstation 44
- Intel i3-7020U @ 2.3 GHz
- 20 GB RAM
- Intel HD Graphics 620 (integrated, no dedicated VRAM — shares system memory via DVMT, up to 32 GB addressable)
- 193.6 GB free disk

All heavy processing was moved to the Jetson. The Fedora machine was used for script development, testing, and verification of small frame batches.

## Input

iPhone 16 Pro Max recording, `.MOV`, 3840×2160, HEVC, 59.97 fps, ~53 seconds.
Frame shows a multimeter (DMM) and an iPhone running the iOS Stopwatch app on a white board. Camera drifts by roughly +78 px horizontal and −30 px vertical across the clip.

## Output

`cap_ocr/data_final.txt` — one line per frame:

```
Volts: 3.25 Time: 0:00.00 (0.00)  (frame 1000, t=16.667s)
```

Fields: DMM reading, timer reading in `M:SS.cc`, timer converted to seconds, frame number, video-time in seconds.

Final run: 2147 good / 2167 total (99.1%).

## Workflow

1. Rotate and deskew video, extract frames
2. Trim dead frames (very helpful in terms of saving time)
3. Template-match the DMM across frames to follow camera drift
4. Crop DMM and timer regions
5. Preprocess each crop for OCR
6. Run OCR (detection + recognition)
7. Parse results into text file
8. OPTIONAL: Process the file into .xlsx for visual data (Included in /processedxlsx)

## Environment Setup (Jetson)

```bash
sudo apt update
sudo apt install -y imagemagick python3-pip python3-venv git make gcc \
    libimlib2-dev libva-utils
python3 -m venv ~/ocr_env
source ~/ocr_env/bin/activate
pip install --upgrade pip
pip install pyturboocr opencv-python numpy Pillow
```

Add swap before any build — the ONNX/TensorRT compile is memory-bound: (here I've used 16GB)

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### GPU ONNX Runtime

The stock `pip install onnxruntime-gpu` ships x86_64 wheels only. There is no aarch64+sm_87 prebuilt. Build it from source using the kit at `github.com/straga/jetson-jp7-onnxruntime`:

```bash
git clone https://github.com/straga/jetson-jp7-onnxruntime
cd jetson-jp7-onnxruntime
make all          # base image + compile + smoke tests, ~4 hours
```

Produces `dist/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl`.

Install into the venv:

```bash
source ~/ocr_env/bin/activate
pip uninstall -y onnxruntime
pip install ~/jetson-jp7-onnxruntime/dist/*.whl
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

A harmless warning appears on every ORT session: `GPU device discovery failed: /sys/class/drm/card1/device/vendor`. Ignore it, CUDA will work.

### Docker access

```bash
sudo usermod -aG docker $USER
# log out and back in
docker ps
```

## Video Processing

### Extract frames

```bash
ffmpeg -i discharge.MOV \
  -vf "fps=60,rotate=-35*PI/180:ow=rotw(-35*PI/180):oh=roth(-35*PI/180):c=black,scale=1920:1080" \
  -q:v 2 frames/frame_%04d.png
```

The `rotate=-35*PI/180` handled both the phone's rotation metadata (ignored by ffmpeg with `-hwaccel_output_format vaapi`) and the scene tilt. Adjust the angle to match your own footage.

Trim to the useful range by copying only frames 1000–3166 into a new folder. The `+repage` fix below must be applied before any cropping — the earlier `rotate` step leaves a page offset (`+0+0` vs `-74-246`) that silently breaks all crop coordinates.

```bash
for f in frames/frame_*.png; do
    convert "$f" +repage frames_flat/"$(basename "$f")"
done
rm -rf frames && mv frames_flat frames
```

### Transfer to Jetson (if you did it locally, I did; it took lots of time on my spec; far better on Jetson)

```bash
ssh admin@<jetson-ip> "mkdir -p ~/cap_ocr/frames"
rsync -avh --progress frames/ admin@<jetson-ip>:~/cap_ocr/frames/
```

## Crop Coordinates

Open any frame in GIMP or `feh --info` and read off the pixel bounds of the DMM digits and the timer. Two rectangles needed:

- DMM: 148×78 at `+603+181`
- Timer: 141×61 at `+714+242`

These are specific to my video. For a different recording, measure new coordinates. Do not skip this step, the OCR can produce garbage on the wrong crop.

## Template Tracking

The camera drifts, so a static crop eventually misses. Template matching on the DMM region handles this.

Build a template from frame 1000 (or any clean frame), then for each frame run `cv2.matchTemplate` over a search window and use the best match location to derive the current crop origin. The timer crop is offset from the DMM by a fixed amount.

Match scores stayed between 0.77 and 1.00 across the clip. Below ~0.5, tracking should be considered failed.

## OCR

`pyturboocr` (PP-OCRv6 tiny tier) was used. It runs detection and recognition in two ONNX sessions.

**Important**: feeding a tight crop directly to the recognition model can produces garbage. Detection must run first to isolate text boxes within the crop. I tried skipping detection but failed.

### Preprocessing (DMM)

```python
r, _, _ = pil_rgb.convert("RGB").split()
arr = np.array(r).astype(np.float32)
lo, hi = 0.30 * 255, 0.70 * 255
arr = (arr - lo) / (hi - lo) * 255
arr = np.clip(arr, 0, 255).astype(np.uint8)
img = Image.fromarray(arr).resize((w*6, h*6), Image.LANCZOS)
```

Red channel only, fixed level stretch 30%–70%, 6× upscale. Fixed levels matter, percentile-based adaptive thresholds produce inconsistent results across frames.

### Preprocessing (timer)

Plain 4× upscale, no channel work. The iOS stopwatch is high-contrast white-on-black and works with a simpler recipe.

### Retry on failure

When the primary preprocessing returns empty text, retry with:

1. Red channel, 20%–80%
2. Red channel, 40%–60%
3. Green channel, 30%–70%
4. Grayscale, 25%–75%

This lifted accuracy from 86% to 99%. Each fallback costs ~10 ms (PIL only).

## Parallelization

Python's GIL blocks true threading for CPU-bound work. `multiprocessing.Pool` with one `OCR()` instance per worker gives real parallelism.

Worker counts tested on the Jetson:

| Workers | Rate | Notes |
|---|---|---|
| 2 | 4.6 fps | underutilized |
| 4 | 8.3–8.8 fps | optimal |
| 6 | 7.6 fps | contention |
| 8 | 6.6 fps | CPU saturated |

Batched inference (feeding N images to a single session call) was also tried. It was 2.0 fps with 73% accuracy, worse on both axes. The model is latency-bound, not throughput-bound; batching adds queueing overhead without GPU utilization gains.

## Run

```bash
source ~/ocr_env/bin/activate
cd ~/cap_ocr
nohup bash -c 'FRAMES_DIR=~/cap_ocr/frames \
  OUTPUT_FILE=~/cap_ocr/data_final.txt \
  START_FRAME=1000 END_FRAME=3166 NUM_WORKERS=4 \
  python3 batch_ocr_parallel_retry.py' \
  > /dev/null 2> ~/cap_ocr/progress_final.log &
```

Monitor:

```bash
tail -f ~/cap_ocr/progress_final.log
```

## Performance

| Configuration | Time (2167 frames) | Rate |
|---|---|---|
| Fedora i3-7020U, single-threaded CPU | 22 min | 1.25 fps |
| Jetson CPU, threaded | 10 min | 3.5 fps |
| Jetson GPU, single process | 15 min | 2.4 fps |
| Jetson GPU, parallel ×4 | 5 min | 7.3 fps |
| **Jetson GPU, parallel ×4, retry** | **4 min** | **8.8 fps** |

The bottleneck is per-inference host overhead, not compute. Scaling further would require batching the whole pipeline (I/O + preprocess + inference) as one job, or a larger model that can actually saturate the GPU.

## Notes

- The `+repage` fix is non-obvious and cost hours. Any ImageMagick operation following a `-rotate` will inherit page geometry offsets that silently corrupt all subsequent crops.
- Template matching fails when the template feature is too small or low-contrast. The DMM digits worked because the LCD pattern is distinctive.
- `pyturboocr`'s `recognize_image()` requires a file path, not a PIL/numpy array. Save to a temp file before each call.
- Onnxruntime on Jetson prints the device-discovery warning on every session. It is cosmetic and does not indicate a real failure.
- The GPU build took ~4 hours and, in practice, delivered ~2× over CPU. For a small model like PP-OCRv6 tiny, CPU is often within a factor of two of GPU. The parallelism between processes is what provides the real speedup, not the GPU itself.

## Repo layout

```
README.md
data/
  <spreadsheets>.xlsx         # analysis outputs, not part of the pipeline
cap_ocr/
  README.md                   # file-by-file description
  batch_ocr_parallel_retry.py # main OCR script
  data_final.txt              # final output, 2147/2167 good
  progress_final.log          # run log
  frames/                     # one sample frame only (full set regenerable)
  ... (other scripts and test outputs)
```

## Reprocessing a new video

1. Extract frames from `.MOV` (ffmpeg one-liner above)
2. Trim to useful range (this step can be very helpful in terms of time)
3. Apply `+repage` to all frames
4. Measure crop coordinates for the DMM and timer in GIMP
5. Update `DMM_W, DMM_H, SEARCH_X, SEARCH_Y, TEMPLATE_CROP` in the script
6. Build a template from a clean frame
7. Run
