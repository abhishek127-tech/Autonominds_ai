# Model Card: FloodSense AI (SegFormer-B0 SAR)

## Overview

- **Planned architecture:** SegFormer-B0 adapted for two-channel VV/VH Sentinel-1 SAR input; this architecture is not implemented in the current code.
- **Current implementation:** `baseline-threshold-stub`, which loads no neural network or checkpoint and emits an all-background mask for readable images.
- **Task:** Semantic flood segmentation.
- **Current version:** M0 stub / baseline.

## Intended Use

The planned system is intended to detect water bodies and flood extents from Sentinel-1 SAR imagery, with research and emergency-monitoring workflows as possible applications after appropriate training and validation. The current stub is only suitable for exercising the file interface and output format. It must not be used for flood decisions or emergency response.

## Limitations and Known Failure Cases

- **Stub limitation:** The current version does not perform neural-network inference or detect water.
- **SAR speckle noise:** Speckle may affect classification boundaries; appropriate preprocessing and validation are still needed.
- No trained checkpoint, documented training data, or labeled evaluation set is included.
- The inference configuration is descriptive and is not consumed by `predict.py`.
- Georeferencing and real-world area calculations are not implemented.
- No performance scores are available; see `metrics.json`.

## Evaluation

Evaluation requires a documented checkpoint, a held-out labeled dataset, and a reproducible preprocessing and metric protocol. The planned dataset is Sen1Floods11; it has not been used to evaluate this stub.