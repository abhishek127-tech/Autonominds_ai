# Model Card

## Model details

The current implementation is a baseline API stub named `baseline-threshold-stub`. It does not load a neural network or checkpoint. For every readable image it emits an all-background mask.

## Intended use

The stub is suitable only for exercising the file interface and output format during development. It is not suitable for flood detection, scientific evaluation, emergency response, or operational mapping.

## Performance

No validation results are available. See `metrics.json`; IoU and pixel accuracy are intentionally `null` rather than fabricated.

## Limitations

- No trained weights or training pipeline are included.
- The inference configuration is descriptive and is not read by `predict.py`.
- Raster georeferencing and real-world area calculations are not implemented.
- Results must not be interpreted as verified flood maps.

## Future evaluation

Evaluation requires a documented checkpoint, a held-out labeled dataset, and a reproducible preprocessing and metric protocol.