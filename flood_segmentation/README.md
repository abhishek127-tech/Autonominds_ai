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

## Sen1Floods11 training data

The training setup reads Sentinel-1 chips from `training/train_images/` and masks from `training/train_masks/`. The configured sample IDs are listed in `training/train_list.txt`; the corresponding files must be named `<ID>_S1Hand.tif` and `<ID>_LabelHand.tif`. Set `SEN1FLOODS11_IMAGE_DIR` or `SEN1FLOODS11_MASK_DIR` to override either directory. Data files are not included in this repository; supply data you are licensed to use. To use the standard Sen1Floods11 folder layout, point the image and mask overrides at `S1Hand/` and `LabelHand/`.

Verify sample membership in the intended training split and follow Sen1Floods11's terms before using or redistributing the data. The training script loads paired two-band SAR and label TIFFs, trains SegFormer-B0, validates each epoch, and saves the best validation-loss checkpoint.

## Project contents

- `configs/` contains the intended class map and inference defaults.
- `examples/` contains the manifest and locations for licensed sample inputs and references.
- `metrics/` records evaluation status and documents the intended dataset and model limitations.
- `model/best_model/` is reserved for a trained checkpoint; none is shipped.
- `training/train.py` contains the training and validation loop; it requires locally supplied TIFF data and the dependencies in `requirements.txt`.
