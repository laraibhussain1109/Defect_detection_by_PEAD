# Validation record

Validated in the managed Linux CPU environment on 2026-11-02.

- 10 pytest checks passed: operator UI/API availability, explicit missing-camera failures, safe image upload/decoding, cross-origin command rejection, native 4K capture storage, complete overlapping tile coverage, ROI bounds, glare exclusion, region localization/area filtering, fixed threshold decisions, and capture-age staleness.
- Desktop (1500 px) and mobile (390 px) Chromium checks passed with no JavaScript page errors or horizontal mobile overflow. Screenshots were visually inspected in the development environment.
- Both official anomalib models trained for one epoch using 24 synthetic normal source images, with six distinct source images held out. EfficientAD's penalty dataloader used a separate synthetic test fixture. This intentionally exercises integration without downloading the production ImageNette dataset.
- Official EfficientAD pretrained teacher weights and hash-checked torchvision ResNet18 weights downloaded successfully. PatchCore coreset fitting, both checkpoint saves, checkpoint reload, full-tile 256×256 maps, map calibration/fusion, scratch-sensitive output, and annotated JPEG generation passed.
- The real saved model pair passed a live HTTP inspection lifecycle test with a simulated camera source: start, background inference, result polling, JPEG preview, and stop.
- Installed dependency consistency (`pip check`) and JavaScript syntax checks passed.

Not validated: physical Makose camera access, Windows DirectShow negotiation, NVIDIA CUDA execution/performance, sustained 4K capture or inference rates, real aluminium defects, 1 mm recall, production false-reject rate, optical setup, and real ImageNette training. The synthetic run is an integration check, not an accuracy benchmark or usable part model.

The cloud proxy currently denies `s3.amazonaws.com`, which hosts the official ImageNette dataset. A draft network addition plus reusable cloud install/start instructions were saved; saving does not apply or publish that change. Windows training needs normal Internet access to that host. Production setup always uses the real ImageNette dataset; the smoke fixture is isolated outside the application's default data directory.

## High-resolution training fix

The original 4,000-tile staging limit rejected a normal 100-image 4K dataset (80 training images × 60 tiles = 4,800 tiles). It is replaced with deterministic source-balanced budgets, with source-level train/validation separation retained.

Additional validation:

- Actual staging regression with 100 distinct 3840×2160 JPEG images plus one exact duplicate: deduplicated to 100 sources; selected 512 of 4,800 training tiles and 128 of 1,200 validation tiles; native 512×512 crops retained; no source overlap across splits.
- Deterministic balanced quotas, spatial coverage, small datasets, budgets below source count, mixed-resolution rejection, legacy job defaults, quick-check overrides, API job serialization, budget bounds, quick-model labeling, and 4K/256 full-frame grid coverage are covered by tests.
- A sampled real-model CPU run completed with 30 synthetic 512×512 sources: 24/216 training tiles and 6/54 validation tiles, one EfficientAD epoch, and PatchCore coreset fitting. Both checkpoints and calibration were saved and reloaded successfully.
- Full-frame tiled predictions and live HTTP inference passed with that sampled model pair and a simulated camera. Inspection cancellation between tiles now permits clean shutdown, including when inference is still processing a frame.
- The updated UI passed mocked-API Chromium checks for quick/normal payloads, budget values, button state, quick-model labeling, JavaScript errors, and mobile layout.

These are regression and integration checks. Physical Windows/NVIDIA camera performance and real defect accuracy still require target-workstation validation.
