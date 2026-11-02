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

## Windows startup and green-backdrop part masking (0.1.x)

The supplied failure log reaches EfficientAD workspace creation, then fails with `WinError 1314` when anomalib requests a `latest` symbolic link. The application now uses anomalib's ordinary, non-versioned workspace inside each already-unique run folder. A regression test denies all symlink creation: stock anomalib reproduces the privilege failure and the application wrapper succeeds without that privilege.

The supplied log also shows global Python 3.13. The pinned model stack supports Python 3.10–3.12; Windows install/start scripts now verify and use the local Python 3.11 environment. Worker runtime validation runs before dataset staging. PowerShell/Windows execution is not available in this Linux environment; the launchers were reviewed, not executed on Windows.

Green masking is shared by training and inference. It neutralizes border-connected green, skips background-only tiles, saves per-tile masks, calibrates on part pixels only, and excludes background from final anomaly decisions. Isolated internal green-visible holes remain part pixels. Blank green views yield `NO_PART`, not `NORMAL`. The preview API can use either the camera or a dataset image.

Validation completed:

- **34 tests passed**, including the Windows privilege reproduction/fix, Python 3.13 early rejection, part-only calibration, background-score suppression, internal 13-pixel holes, native-resolution masked staging, `NO_PART`, preview API, and prior 4K sampling regressions.
- Both real models trained and reloaded successfully with a synthetic 1024×1024 green-backed part, using 24 training tiles and six held-out tiles. Cached official pretrained weights and a synthetic test-only penalty-image fixture were used.
- Full-frame prediction and simulated live HTTP inference passed. Changing only the excluded green backdrop's brightness left predictions unchanged. A synthetic enclosed green-visible hole changed the model maps, while all excluded background scores remained zero. This is an integration check, not real 1 mm dent/hole recall evidence.
- Mocked-API Chromium tests passed for mask controls/defaults, recipe restoration, preview request/response/error handling, and mobile layout. JavaScript syntax and installed dependency checks passed.

At the provided 298×165 mm field of view and 3840×2160 capture, 1 mm occupies approximately 13 camera pixels. New-job 256-pixel tiles preserve that nominal density at the model input; actual shallow-dent/blowhole sensitivity remains unmeasured until tested on real defect samples with the physical camera and lighting.


## Python 3.13 support (0.2.0)

This release supersedes the earlier Python 3.13 restriction. The dependency stack was migrated, not just the interpreter guard: anomalib 2.7.0, PyTorch 2.10.0, torchvision 0.25.0, and NumPy 2.x. The outdated mandatory OpenVINO dependency was removed. The Windows installer now prefers the existing standard 64-bit Python 3.13, with supported 3.12/3.11 fallback, and installs the matching CUDA 12.6 or optional CPU wheels.

Validated in a clean **CPython 3.13.16** environment:

- Fresh dependency installation completed, actual EfficientAD/PatchCore imports passed without OpenVINO, and `pip check` reported no broken requirements.
- **42 tests passed**, including Python 3.13 acceptance, old-stack diagnostics, symlink-free workspace setup, CPU/CUDA build tags, Windows interpreter-build checks, all part-mask and 4K sampling regressions, and safe rejection of legacy checkpoints before pickle loading.
- Both official models completed actual CPU training on the isolated green-background synthetic fixture (24 training / six held-out tiles), calibrated on part pixels, saved new-format checkpoints, and reloaded them with `weights_only=True`.
- Full-frame tiled inference and simulated live HTTP inspection passed. Excluded-background brightness changes left maps unchanged; a synthetic enclosed hole changed predictions; ignored-background scores remained zero. The live loop stopped cleanly.
- Exact official **Windows CPython 3.13 CUDA 12.6 wheels** for torch 2.10.0 and torchvision 0.25.0 downloaded successfully. This verifies binary availability, not execution on a physical NVIDIA device.
- JavaScript syntax and whitespace checks passed. PowerShell scripts were reviewed, not executed on Windows; physical Windows/CUDA/camera behavior and real defect recall remain target-workstation checks.

Checkpoint format 2 uses the updated stack's safe plain-data component configuration. Format 1 model files are retained but require retraining; dataset/source images are not modified or removed. The synthetic penalty-image fixture is test-only, as in prior integration checks.

## Windows long-path installer recovery (0.2.1)

The reported package-install failure occurred under a 153-character checkout root. The installer had put `.venv` under that root, so setuptools' nested package files exceeded the legacy Windows path limit. Installation stopped before the required anomalib version could be installed.

The installer and launcher now share `scripts/windows-paths.ps1`. Dependencies live under `%USERPROFILE%\.pead\v\<project-id>`, not inside the checkout. Temporary wheel builds and model assets use short runtime paths too; an already-existing short pip cache is reused to avoid re-downloading large wheels. Existing local `.venv` files and all source images remain untouched. Incomplete managed environments are preserved before rebuilding.

A readiness marker is written only after package installation, `pip check`, and actual model imports succeed. The launcher checks that marker's project and dependency hash and refuses incomplete installations; it never falls back to checkout-local or global Python.

Validation:

- Executed the actual installer, shared resolver, and launcher under **PowerShell 7.4.6** on Linux, using isolated Python/pip command fixtures, with **53 passing assertions**. PowerShell itself was downloaded from its official release and checked against the published SHA256 before use.
- The harness runs from a checkout at least 153 characters long and reconstructs the reported setuptools suffix: the old checkout-local target exceeds 260 characters; the managed target stays below it. It verifies matching install/start interpreter paths, source/partial-environment preservation, cache reuse, overrides, manifest changes, and path guards.
- Injected failures at PyTorch installation, application dependency installation, dependency verification, and runtime imports. Each failure revoked readiness and prevented server launch; repair completed and restored readiness.
- Inspected the actual Windows CPython 3.13 PyTorch 2.10.0 CUDA wheel. Its longest package member is 119 characters; with the default short runtime path for the reported user profile, the installed target is approximately 172 characters.
- Python regressions also exercise the external model-assets override without moving dataset/run paths. Dependency consistency, JavaScript syntax, and whitespace checks passed.

These are real PowerShell script-execution tests with controlled native-command fixtures and Windows-wheel path analysis. They do not claim an actual pip/CUDA installation on Windows or execution on the user's physical PC. The repair avoids the known long-path source rather than changing Windows registry policy.
