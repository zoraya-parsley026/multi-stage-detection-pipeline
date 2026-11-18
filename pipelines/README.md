# Running the pipelines

Run these commands from the repository root, using the project's virtual environment:

```bash
.venv/bin/python pipelines/yolov7_pipeline.py --no-progress
.venv/bin/python pipelines/yolov9_pipeline.py --no-progress
.venv/bin/python pipelines/efficientformer_pipeline.py --no-progress
.venv/bin/python pipelines/thesis_pipeline.py --no-progress
```

Module execution also works, for example `python -m pipelines.yolov7_pipeline`.
Use `--help` on any script for its options. Default data, configuration, and weight
paths are relative to the working directory. The ONNX weights must be available
locally in `weights/`.

`--images` accepts a directory, a single image, or a quoted glob. For a small check:

```bash
.venv/bin/python pipelines/thesis_pipeline.py \
  --images in_data/subset/images/000000002261.jpg \
  --labels in_data/subset/labels \
  --output runs/results/thesis_sample.json \
  --plots-dir runs/plots/sample --no-progress
```

The detector scripts evaluate predictions against YOLO labels and write metric
reports to JSON. `--conf` controls prediction confidence filtering; `--iou` controls
NMS suppression, not the evaluation matching threshold.

The EfficientFormer script classifies whole images and writes prediction counts
and mean confidence. It does not measure classification accuracy against detection
labels. `images_processed` counts all images; `images_classified` counts those with
a returned prediction. Mean confidence and class counts use returned predictions.

The thesis script evaluates a detector-only baseline, then a separate cascade run
with cropping, classification, class translation, and filtering. The detector runs
twice so each report measures its own pipeline runtime. Timings exclude model
loading and image loading; there is no warm-up or repeated benchmark averaging.
The translation map must exist. Reports go to `runs/results/` and five comparison
plots go to `runs/plots/` by default.

Run tests with:

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q --run-integration
```

The first command skips local model integration tests. The second enables them;
tests whose model weights or reference files are missing are still skipped.
Pipeline script regression tests use small synthetic images and fake models to
check report output, baseline timing, empty predictions, and plot generation.
