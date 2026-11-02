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
