# PEAD inspection workstation

A local Windows-first workstation combining **official anomalib EfficientAD-small and PatchCore (ResNet18)** for visual anomaly detection, trained on good images only. Camera capture, dataset folder import, training, and live overlays are available in one UI. This is an engineering prototype with production-oriented safeguards; acceptance on your actual parts remains required.

## Windows + NVIDIA installation

Install **Python 3.11 x64**, an up-to-date NVIDIA driver supporting CUDA 12.1, and Git. A CUDA toolkit installation is not required for the PyTorch wheels. Use an NVIDIA GPU with at least 8 GB VRAM as a starting point; memory and speed depend on dataset, tile count, and PatchCore coreset size.

From PowerShell in this repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Open the local application at `http://127.0.0.1:8000` on that same computer. Keep the server bound to loopback: there is no multiuser authentication. Plug the Makose 8.3 MP camera into the **Windows computer running the server**, select its device index, and connect. A cloud machine cannot access a USB camera attached to your PC. Close other camera apps first. Windows capture uses DirectShow, requests MJPG / 3840×2160 / 30 FPS, and reports negotiated resolution and FPS; the driver may select a different mode. A USB 3 port and compatible cable are recommended. Capture and inference rates are separate; 4K at 30 capture FPS is not a claim of 30 inference FPS.

For CPU development on Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[ml,dev]'
.venv/bin/python -m uvicorn pead.app:app --host 127.0.0.1 --port 8000
```

Python 3.10–3.12 is supported. First training needs Internet for EfficientAD teacher weights (GitHub release), hash-checked PatchCore backbone weights (PyTorch), and the official ImageNette penalty dataset (`s3.amazonaws.com`, about 1.5 GB). Dependencies use PyPI. All downloads retain TLS and upstream verification. Training errors appear in the UI log; a failed run does not replace the active model.

## Windows training error and supported interpreter

`WinError 1314` in anomalib's workspace setup was caused by a directory symlink requiring a Windows privilege. Training now uses ordinary run-specific directories and does not require administrator rights or Developer Mode for symlinks.

Use the provided installation and startup scripts. Python **3.13 is unsupported** by this pinned model stack. A traceback under `Python313/Lib/site-packages` means the application was started using that global interpreter rather than the prepared environment. Run `scripts/install-windows.ps1` with Python3.11 installed, then `scripts/start.ps1`; the launcher checks the local virtual environment and dependency versions. An incompatible existing `.venv` is preserved as a backup before a supported environment is created. Keep your `data` folder and source images.

## Operator workflow

1. Fix camera, lens focus, working distance, part fixture, exposure, white balance, and lighting. Camera exposure/white balance should be locked through the vendor's controls; this UI does not assume universal UVC exposure controls.
2. Keep **Ignore green background** enabled when the part is on a green backdrop. Click **Preview part mask** to check the latest camera frame or the first image in your selected dataset. The complete part must be included. Set a normalized rectangular ROI to restrict the working area further when needed. Use the same part orientation and camera resolution for every image, and a separate dataset/model for each part and view.
3. Capture verified good parts in the UI, select a folder to upload, or enter an existing server-local dataset folder. Empty path uses `data/captures`. Imported folders are copied into that dataset. Use a different `PEAD_DATA_DIR` for a separate part to avoid mixing captures.
4. Collect at least **30 distinct good images**, preferably 100–500 representing acceptable finish, machining marks, and lighting variation. Directory scanning is recursive. Supported formats: PNG, JPEG, BMP, TIFF. Exact byte duplicates are removed before the seeded 80/20 source-image split. Near duplicates from the same physical part/session can still bias validation: collect distinct parts and sessions. Holdout images never enter either model's training memory.
5. Verify the part mask, then choose tile size, minimum region area, optional CLAHE, glare limit, and tile budgets. Use 256-pixel tiles to preserve more native detail for small defects when your optics provide enough pixels; this increases inference work. To check the workflow first, press **Quick check · 1 epoch**: it trains both real models using at most 128 training tiles, 64 validation tiles, and a 0.1% PatchCore memory ratio. It preserves the tile size and ROI you selected. Quick-check models are labelled **QUICK CHECK · WORKFLOW TEST** beside the active model. They are suitable for testing training and live inference, not acceptance of real defects. For a normal training run, press **Train EfficientAD + PatchCore**. The default is 100 EfficientAD epochs, a 1% PatchCore coreset, and budgets of 512 training / 128 validation tiles. Epochs are configurable and do not guarantee convergence. PatchCore fits its memory bank in one pass. A smaller coreset trades recall for resources. Up to 2000 source images are supported; every source image must have the same camera resolution.
6. After successful training, press **Start inspection**. Training and inspection are mutually exclusive. Model artifacts include the exact preprocessing recipe; changing UI values does not silently change a loaded model. Retrain to change the recipe. Training can be cancelled; incomplete runs are never activated.

Large datasets no longer fail at 4000 tiles per split. For example, 100 full 4K images with 512-pixel tiles and 25% overlap yield 4800 training tiles before sampling. The source images are deduplicated and split first, then a seeded, source-balanced selection chooses representative tile positions within each budget. Small datasets retain all tiles when they fit. Training and validation remain separate by source; their selected counts and source coverage are printed in the training log, saved in the run's `sampling.json`, and summarized in the model manifest. A budget smaller than the number of source images cannot include every source, so inspect that coverage before accepting a model. Budget limits are 24–1024 training tiles and 6–512 validation tiles. Increasing them requires more training time and PatchCore memory; sampling can miss normal variation and must be validated on your parts.

Sampling applies only to training and calibration staging. It does not resize the source image or reduce the selected tile size. Live inference examines the eligible overlapping part tiles across the ROI, including when the view contains more than 128 tiles. Background-only tiles are skipped when green-background exclusion is enabled.

The live preview uses blue/green for low deviation on the part, yellow for doubtful regions, red for regions above the calibrated anomaly threshold, and magenta for clipped glare. Excluded background stays as raw camera imagery without a heatmap, and the part outline is shown. A missing part gives a separate NO_PART decision. Bounding boxes are drawn on connected regions above a minimum area. NORMAL, REVIEW, ANOMALY, and UNINSPECTABLE are separate decisions. Any detected clipped glare prevents NORMAL. Thresholded glare is a heuristic, not a detector of all specular reflections. Capture saves the raw full-resolution image, not the resized heatmap preview.

## Part mask and green backdrop

**Ignore green background** is enabled for new training jobs in the UI. It uses HSV color thresholds to exclude green areas connected to the border of your rectangular ROI. Isolated green pixels inside the part silhouette are retained, which helps preserve holes or pits revealing the backdrop for anomaly inspection. This is chroma-key masking, so it needs a distinguishable green background; it is not a general object detector. Turn it off for a different backdrop or a green part. Existing saved models retain their original recipe until you retrain.

Click **Preview part mask** before training. It uses a fresh camera frame when available, otherwise an image from the dataset folder (or captured-image dataset when the folder path is empty). The preview dims excluded background and shows the part contour. The displayed percentage is the part fraction of the ROI. Adjust hue limits, minimum saturation, and minimum brightness until the complete inspected surface is included. OpenCV hue values are 0–179, not degrees. Check shadows, part edges, and areas that reflect the green backdrop; reflected green on aluminium can otherwise hide part pixels from inspection.

Training selects only crops containing at least the configured minimum part fraction. Excluded background is replaced by a fixed neutral fill in both training and inference, held-out calibration uses valid part pixels, and anomaly decisions/heatmaps are restricted to the part. This avoids spending the tile budget learning the green sheet. Glare handling still applies to the inspected surface. Mask settings are saved with the model, so changing them requires retraining. Masking does not recover a defect hidden by glare, blur, or insufficient optical detail.

## Model and calibration

EfficientAD uses its pretrained PDN teacher, learned student and autoencoder, plus ImageNette penalty images. PatchCore uses pretrained ResNet18 layer2/layer3 patch embeddings and a sampled normal-feature memory bank. This is an OR-style hybrid: each raw map is scaled against its own held-out good-image median and 99.95th percentile, then the pixelwise maximum is fused. These are **normal-only deviation thresholds, not probabilities or measured defect accuracy**. Yellow starts at 0.65 of the calibrated scale and red at 1.0. Minimum-area filtering suppresses tiny connected components. Tile overlap is 25%; overlapping outputs fuse by maximum. Each tile is resized to the models' default 256×256 input. New UI/API jobs default to 256-pixel native tiles and a four-pixel minimum connected region, preserving more fine detail than the former 512-pixel/20-pixel defaults. Existing saved recipes keep their original settings; the smaller region threshold can increase noise and must be checked on independent good parts. PatchCore is a throughput bottleneck; reducing coreset size reduces memory and nearest-neighbor cost. Conservative max fusion can increase false positives.

Images are processed at their native resolution for ROI and tiling. Inference drains the latest camera frame rather than queueing every frame; frames are intentionally dropped when inference is slower than capture. Preview is limited to 1440 pixels wide. The UI reports inference latency and marks decisions stale when their input frame is older than two seconds, including time spent in inference. This is suitable for stationary or indexed parts; moving conveyor parts need trigger synchronization, exposure control, and motion-blur validation, which are not implemented here.

## Detecting 1 mm defects

Megapixels alone do not establish detection capability. Measure the **physical horizontal field of view** at the part surface. Native pixels per mm = camera width / field-of-view width in mm. Approximate model pixels across a 1 mm defect = native pixels/mm × 256 / tile size. For a 3840-pixel image spanning 400 mm, a 512-pixel tile gives about **4.8 model pixels/mm**; a 1024-pixel tile gives only 2.4. Prefer at least roughly 5–10 model pixels across the smallest target as an initial optical design goal, then validate experimentally. Contrast, scratch width, focus, perspective, motion blur, sensor noise, and model feature stride matter. A 1 mm-long hairline scratch may still be far narrower than one model pixel. Reduce field of view, reduce tile size, or add views as needed. The minimum-area setting is in ROI camera pixels, not square millimetres.

Build a separate acceptance set with known 1 mm defects and independent good parts. Check misses, false rejects, localization, and latency at the actual line speed, lighting, and fixture tolerances. Normal-only training cannot establish recall. There is no certified pass/fail guarantee, PLC reject control, segmentation ground-truth evaluation, or recipe approval system yet.

## Aluminium glare and reflection

Use a diffuse dome or large-area light, angle lighting away from the specular return, block uncontrolled ambient light, and test cross-polarized illumination plus a lens polarizer. Bare aluminium may not respond to cross-polarization as well as dielectric surfaces; geometry and diffusion remain critical. Lock exposure without clipping highlights. CLAHE can reduce local brightness variation but can also amplify texture/noise. It is optional, off by default, and applied identically during staging and inference. The application does **not** inpaint glare or blur away scratches. Clipped regions are magenta and excluded from anomaly decisions; a high clipped fraction makes the entire result UNINSPECTABLE. Non-clipped reflections can still trigger anomalies or obscure defects.

## Storage and operations

`PEAD_DATA_DIR` defaults to `./data`:

- `captures/`: full-resolution good captures and imported images.
- `models/<run>/`: immutable job settings, staged train/validation tiles, both checkpoints, calibration manifest, result, and log.
- `models/active.json`: atomically published reference to the last verified complete pair.
- `inspection.jsonl`: per-inference timestamp, model ID, decision, glare fraction, and ROI-relative bounding boxes.

Source datasets are read-only. Part masks for staged validation tiles are saved separately from the image folders so they cannot accidentally be learned as normal images. Checkpoints are local trusted PyTorch artifacts; do not load checkpoints from untrusted sources. Use one server process (no multiple Uvicorn workers or hot reload around a live camera). Stop the server before deleting old runs. Logs are local and unbounded: archive/rotate `inspection.jsonl` and clean retired staged datasets periodically. Disk exhaustion is reported as an error. There is no automatic line-stop integration or backup service.

## Development verification

```bash
python -m pytest
python -m pead.smoke --directory /absolute/path/to/disposable-smoke-data
```

The second command downloads real teacher/backbone weights and trains both official models for one epoch on generated good images with a generated penalty-image fixture. It validates plumbing, checkpoint reload, heatmap generation, and the HTTP live-inference lifecycle with a simulated camera; it is **not** a defect accuracy benchmark and does not use the real ImageNette distribution. Run it in a separate data directory; it does not replace the operator dataset. Windows GPU/camera checks and real defect acceptance must be run on the target workstation.

To exercise representative crop sampling in the disposable integration smoke:

```bash
python -m pead.smoke --directory /absolute/path/to/new-sampling-smoke --image-size 512 --train-tile-budget 24 --validation-tile-budget 6 --quick-check
```

This selects 24 of 216 training crops and 6 of 54 held-out crops, then checks saved-model inference with a simulated camera. Test images and the penalty-image fixture are generated only inside the disposable directory.

For the reported **298 mm × 165 mm field of view at 3840×2160**, the scale is about **12.9 horizontal / 13.1 vertical camera pixels per mm**. The new-job 256-pixel tile setting keeps that native pixel density at the 256×256 model input; a nominal 1 mm hole spans about 13 input pixels. The four-pixel minimum region avoids discarding such regions solely by area. These settings preserve the available detail; they do not measure recall for shallow dents, blowholes, or low-contrast scratches. Validate those defect types with real samples under the fixed inspection lighting.

The green mask is a chroma key, not a learned semantic segmentation model. Non-green fixtures and shadows can remain in the part mask; use the rectangular ROI and mask preview to exclude them. Material loss connected to the outer green background can alter the silhouette and be masked out: this surface-anomaly workflow does not replace a reference silhouette/geometry inspection.
