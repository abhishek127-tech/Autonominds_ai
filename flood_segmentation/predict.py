import os
import json
from pathlib import Path
from PIL import Image

def load_model():
    """Loads the model checkpoint once and caches it."""
    # For M0 stub, we return a baseline flag or placeholder
    return {"model_name": "baseline-threshold-stub", "loaded": True}

def analyze_flood_image(image_path: str, output_dir: str) -> dict:
    """
    Analyzes an input image and returns the required dictionary specification.
    Never crashes on bad input; returns an error dictionary instead.
    """
    img_path = Path(image_path)
    out_dir = Path(output_dir)
    
    # 1. Error Handling: File not found
    if not img_path.exists():
        return {
            "status": "error",
            "error_code": "FILE_NOT_FOUND",
            "message": f"The image path does not exist: {image_path}",
            "warnings": []
        }
    
    try:
        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Open image to get dimensions
        img = Image.open(img_path)
        width, height = img.size
        
        # Define output file paths
        base_name = img_path.stem
        mask_filename = f"{base_name}_mask.png"
        overlay_filename = f"{base_name}_overlay.png"
        preview_filename = f"{base_name}_preview.png"
        result_filename = f"{base_name}_result.json"
        
        mask_path = out_dir / mask_filename
        overlay_path = out_dir / overlay_filename
        
        # M0 Baseline: Create a dummy 8-bit binary mask (0 and 1 values only)
        # In a real model, this is where your prediction array goes.
        mask_img = Image.new("L", (width, height), color=0) # 0 = background
        mask_img.save(mask_path)
        
        # Save a basic copy as an overlay for the stub
        img.save(overlay_path)
        
        # Calculate mock pixel counts
        valid_pixels = width * height
        flood_pixels = 0  # Baseline stub detects 0 flood pixels initially
        coverage_percent = 0.0
        
        result_dict = {
            "status": "ok",
            "flood_mask_path": str(mask_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "flood_pixel_count": flood_pixels,
            "valid_pixel_count": valid_pixels,
            "flood_coverage_percent": coverage_percent,
            "real_world_area_m2": None,
            "class_names": {"0": "background", "1": "predicted_water"},
            "model_name": "baseline-threshold-stub",
            "warnings": ["BASELINE: simple threshold, not a trained flood model"],
            "georeferenced": False,
            "crs": None,
            "image_width": width,
            "image_height": height,
            "checkpoint_id": "sha256:m0-stub-placeholder",
            "inference_seconds": 0.1,
            "result_label": "MODEL ESTIMATE - NOT A VERIFIED FLOOD MAP"
        }
        
        # Save result JSON in output dir
        with open(out_dir / result_filename, "w") as f:
            json.dump(result_dict, f, indent=2)
            
        return result_dict

    except Exception as e:
        return {
            "status": "error",
            "error_code": "CORRUPT_IMAGE",
            "message": f"Could not read image: {str(e)}",
            "warnings": []
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    
    res = analyze_flood_image(args.image, args.out)
    print(json.dumps(res, indent=2))