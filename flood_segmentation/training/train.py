"""Train SegFormer-B0 for two-channel Sentinel-1 flood segmentation."""

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.enums import Resampling
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
IMAGE_DIR = Path(os.environ.get("SEN1FLOODS11_IMAGE_DIR", TRAINING_DIR / "train_images"))
MASK_DIR = Path(os.environ.get("SEN1FLOODS11_MASK_DIR", TRAINING_DIR / "train_masks"))
TRAIN_LIST = TRAINING_DIR / "train_list.txt"
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "best_model" / "pytorch_model.pt"
IGNORE_INDEX = -1
SAR_MIN_DB = -35.0
SAR_MAX_DB = 5.0


class FloodDataset(Dataset):
    def __init__(self, sample_ids: list[str], image_size: int = 512) -> None:
        self.image_size = image_size
        self.samples = []

        for sample_id in sample_ids:
            image_path = IMAGE_DIR / f"{sample_id}_S1Hand.tif"
            mask_path = MASK_DIR / f"{sample_id}_LabelHand.tif"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing image/mask pair for {sample_id}: "
                    f"{image_path} and {mask_path}"
                )
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        output_shape = (self.image_size, self.image_size)

        with rasterio.open(image_path) as source:
            if source.count != 2:
                raise ValueError(
                    f"Expected two VV/VH bands in {image_path}, found {source.count}"
                )
            image = source.read(
                out_shape=(2, *output_shape),
                resampling=Resampling.bilinear,
            ).astype(np.float32, copy=False)
            if source.nodata is not None:
                image = np.where(image == source.nodata, SAR_MIN_DB, image)

        image = np.nan_to_num(
            image,
            nan=SAR_MIN_DB,
            posinf=SAR_MAX_DB,
            neginf=SAR_MIN_DB,
        )
        image = np.clip(image, SAR_MIN_DB, SAR_MAX_DB)
        image = (image - SAR_MIN_DB) / (SAR_MAX_DB - SAR_MIN_DB)

        with rasterio.open(mask_path) as source:
            mask = source.read(
                1,
                out_shape=output_shape,
                resampling=Resampling.nearest,
            )

        mask = np.rint(mask).astype(np.int64, copy=False)
        invalid_values = set(np.unique(mask).tolist()) - {IGNORE_INDEX, 0, 1}
        if invalid_values:
            raise ValueError(f"Unexpected mask values in {mask_path}: {invalid_values}")

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask))
        return image_tensor, mask_tensor


def build_model(device: torch.device) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        id2label={0: "background", 1: "predicted_water"},
        label2id={"background": 0, "predicted_water": 1},
        ignore_mismatched_sizes=True,
    )

    old_conv = model.segformer.encoder.patch_embed1.proj
    if old_conv.in_channels < 2:
        raise ValueError("The pretrained patch embedding has fewer than two channels")

    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight[:, :2])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.segformer.encoder.patch_embed1.proj = new_conv
    return model.to(device)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    valid_pixel_count = 0
    correct_pixel_count = 0
    flood_intersection = 0
    flood_union = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if not torch.any(masks != IGNORE_INDEX):
            raise ValueError("Batch contains no labeled pixels")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = criterion(logits, masks)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

        predictions = logits.argmax(dim=1)
        valid_pixels = masks != IGNORE_INDEX
        valid_pixel_count += int(valid_pixels.sum().item())
        correct_pixel_count += int(
            ((predictions == masks) & valid_pixels).sum().item()
        )
        predicted_flood = (predictions == 1) & valid_pixels
        actual_flood = (masks == 1) & valid_pixels
        flood_intersection += int((predicted_flood & actual_flood).sum().item())
        flood_union += int((predicted_flood | actual_flood).sum().item())

    if sample_count == 0 or valid_pixel_count == 0:
        raise ValueError("DataLoader produced no usable samples or labeled pixels")

    return {
        "loss": total_loss / sample_count,
        "pixel_accuracy": correct_pixel_count / valid_pixel_count,
        "flood_iou": flood_intersection / flood_union if flood_union else 0.0,
    }


def train_model() -> None:
    parser = argparse.ArgumentParser(description="Train FloodSense SegFormer-B0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        parser.error("epochs, batch-size, and image-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Masks: {MASK_DIR}")

    with TRAIN_LIST.open(encoding="utf-8") as split_file:
        sample_ids = [line.strip() for line in split_file if line.strip()]
    if len(sample_ids) < 2:
        raise ValueError("At least two sample IDs are required for train/validation split")

    dataset = FloodDataset(sample_ids, image_size=args.image_size)
    validation_size = max(1, round(len(dataset) * 0.2))
    validation_size = min(validation_size, len(dataset) - 1)
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(42)
    training_set, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    training_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        training_metrics = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={training_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_pixel_accuracy={validation_metrics['pixel_accuracy']:.4f} "
            f"val_flood_iou={validation_metrics['flood_iou']:.4f}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best checkpoint to {CHECKPOINT_PATH}")

    print("Training process finished.")


if __name__ == "__main__":
    train_model()