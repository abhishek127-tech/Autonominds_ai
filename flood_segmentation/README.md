# Flood Segmentation

This package provides a small image-analysis interface for flood segmentation experiments.

## Status

The current `predict.py` implementation is a baseline interface stub, not a trained flood model. It writes an all-background mask and reports zero flood coverage. The inference configuration and class labels describe the intended interface but are not loaded by the current predictor. Do not use its output for emergency response, mapping, or other operational decisions.

No model checkpoint or sample dataset is included. Add only data and weights that you are licensed to use.

## Requirements

Use Python 3.11. Create an environment and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run inference

From this directory, run:

```bash
python predict.py --image /path/to/image.png --out outputs
```

The command prints a JSON result and writes a mask, overlay, and result JSON into the output directory. Supported input formats depend on Pillow; georeferencing is not currently preserved.

## Tests

```bash
python -m pytest -q
```

## Project contents

- `configs/` contains the intended class map and inference defaults.
- `examples/` contains the manifest and locations for licensed sample inputs and references.
- `metrics/` records evaluation status and documents the intended dataset and model limitations.
- `model/best_model/` is reserved for a trained checkpoint; none is shipped.
- `training/train.py` is an explicit placeholder until a dataset format and training pipeline are selected.
