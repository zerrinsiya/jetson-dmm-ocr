# cap_ocr

Working directory for the DMM + stopwatch OCR pipeline. Developed on Fedora 44 (i3-7020U), run on a Jetson Orin Nano Super 8GB. Full pipeline writeup, hardware specs, and performance numbers are in the top-level `README.md`.

## Files

| File | Purpose |
|---|---|
| `batch_ocr.py` | Original single-process OCR script. Superseded. |
| `batch_ocr_batched.py` | Attempt to run recognition without detection. Failed — recognition needs detection to isolate text boxes. Kept for reference. |
| `batch_ocr_batched_v2.py` | Batched detection + recognition in one script. 2.0 fps, 73% accuracy. Slower and less accurate than the parallel version. |
| `batch_ocr_fast.py` | Single-process OCR with PIL-based preprocessing (no ImageMagick subprocess). 2.4 fps. |
| `batch_ocr_gpu.py` | Version of `batch_ocr.py` configured for the CUDA execution provider. Superseded. |
| `batch_ocr_parallel.py` | Multiprocess OCR — 4 workers, one `pyturboocr` instance each. 7.3 fps. |
| `batch_ocr_parallel_retry.py` | **Main script.** Same as above with fallback preprocessing on failed reads. 8.8 fps, 99.1% accuracy. |
| `batch_ocr_threaded.py` | ThreadPool version — GPU inference overlapped with CPU preprocessing. 3.5 fps. |
| `data.txt` | First full run output, single-process. Superseded by `data_final.txt`. |
| `data_final.txt` | **Final output.** 2147 good / 2167 total. Format: `Volts: X.XX Time: M:SS.cc (SECONDS) (frame N, t=X.XXXs)`. |
| `frames_COMPRESSED/` | Extracted frames from the source `.MOV`, descaled to 1920×1080 and rotated −35° to match the physical orientation of the scene. 2167 frames total AFTER TRIMMING (originally f3166). (ONLY EVERY 200TH FRAME IS ATTACHED DUE TO MASSIVE SIZE.)|
| `progress_final.log` | stderr log from the final parallel-with-retry run. |
| `progress.log` | Log from an earlier single-process run. |
| `test_batched.txt`, `test_batched_v2.txt`, `test_fast.txt`, `test_gpu.txt`, `test_parallel.txt`, `test_retry.txt`, `test_threaded.txt` | Output snapshots from each pipeline variant, used to compare speed and accuracy. |

## Video input

Source recording: iPhone 16 Pro Max, `.MOV`, 3840×2160 HEVC, 59.97 fps, ~53 seconds. Descaled to 1920×1080 and rotated −35° to level the DMM display in frame. The rotation angle and final resolution were chosen to match my own physical setup; both are adjustable in the ffmpeg extraction command. After rotation the frame canvas is 1192×1080.

Every frame needs a `+repage` pass before cropping. The `-rotate` step in ffmpeg leaves a page-geometry offset baked into the PNG that ImageMagick silently uses as the crop origin, corrupting all downstream crops.

## Crop coordinates

DMM: 148×78 at `+603+181`
Timer: 141×61 at `+714+242`

Specific to my recording. New footage needs new coordinates measured in GIMP, `feh --info`, or any image viewer that reports pixel positions. I won't be explaining those here.

## AI assistance

A significant amount of this pipeline was written with the help of an AI assistant; DeepSeekV4-Pro && Claude-Opus5.5. Public documentation for PP-OCRv6 in Python on a Jetson Orin, batched and multi-process, with a template-tracking front end is essentially nonexistent. The specific combinations of preprocessing parameters, ORT provider settings, worker counts, and retry fallbacks were done thru trial and error rather than reference material. Anyone reproducing this on different hardware or with different source footage should expect to re-tune most of the constants. This is not as easy as a plug-and-play.
