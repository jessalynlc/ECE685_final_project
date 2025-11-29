import os
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Any


def analyze_image_folder(path: str) -> Dict[str, Any]:
    """
    Analyze a folder of images.

    Returns:
        {
            "num_images": int,
            "shape_counts": { (H, W, C): count, ... },
            "pixel_stats": { "min": float, "max": float, "mean": float, "std": float }
        }
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    shape_counts: Dict[str, int] = {}

    global_min = float("inf")
    global_max = float("-inf")
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    num_images = 0

    for fname in os.listdir(path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue
        
        full_path = os.path.join(path, fname)

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            # skip corrupted images
            continue

        arr = np.array(img, dtype=np.float32)
        num_images += 1

        # Count shapes
        shape = str(arr.shape)  # (H, W, 3)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        # Global pixel statistics
        local_min = arr.min()
        local_max = arr.max()

        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

        pixel_sum += arr.sum()
        pixel_sq_sum += np.square(arr).sum()
        pixel_count += arr.size

    # mean and std
    if pixel_count > 0:
        mean = pixel_sum / pixel_count
        var = (pixel_sq_sum / pixel_count) - (mean ** 2)
        std = float(np.sqrt(max(var, 0)))
    else:
        mean = std = 0.0

    return {
        "num_images": num_images,
        "shape_counts": shape_counts,
        "pixel_stats": {
            "min": float(global_min),
            "max": float(global_max),
            "mean": float(mean),
            "std": float(std),
        }
    }


import os
from typing import Dict, Any, Tuple, List
from PIL import Image
import numpy as np


def analyze_image_folder_classwise(
    folder: str,
    labels: Dict[str, List[int]]
) -> Dict[str, Any]:
    """
    Compute class-wise image statistics for a multi-label dataset.

    Args:
        folder (str): Path to folder containing the images.
        labels (Dict[str, List[int]]): Mapping from image filename to multi-hot label vector.

    Returns:
        Dict[str, Any]: Dictionary containing class-wise statistics:
            {
                class_index: {
                    "num_images": int,
                    "shape_counts": { (H, W, C): count, ... },
                    "pixel_stats": {
                        "min": float, "max": float, "mean": float, "std": float
                    }
                },
                ...
            }
    """
    # Determine number of classes from the first label
    if len(labels) == 0:
        raise ValueError("Label dictionary is empty.")

    num_classes = len(next(iter(labels.values())))

    # Initialize per-class accumulators
    stats = {}
    for c in range(num_classes):
        stats[c] = {
            "num_images": 0,
            "shape_counts": {},
            "pixel_sum": 0.0,
            "pixel_sq_sum": 0.0,
            "pixel_count": 0,
            "min": float("inf"),
            "max": float("-inf"),
        }

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    # Process all images from the label dictionary
    for fname, lb in labels.items():
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue

        full_path = os.path.join(folder, fname)
        if not os.path.exists(full_path):
            continue

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        arr = np.array(img, dtype=np.float32)
        shape = str(arr.shape)

        local_min = arr.min()
        local_max = arr.max()
        s = arr.sum()
        s2 = np.square(arr).sum()
        count = arr.size

        # For each class present in this image
        for c, active in enumerate(lb):
            if active != 1:
                continue

            class_stat = stats[c]

            # count images
            class_stat["num_images"] += 1

            # shape counts
            class_stat["shape_counts"][shape] = class_stat["shape_counts"].get(shape, 0) + 1

            # pixel stats
            class_stat["min"] = min(class_stat["min"], local_min)
            class_stat["max"] = max(class_stat["max"], local_max)
            class_stat["pixel_sum"] += s
            class_stat["pixel_sq_sum"] += s2
            class_stat["pixel_count"] += count

    # Finalize mean/std per class
    output = {}
    for c in range(num_classes):
        st = stats[c]
        if st["pixel_count"] > 0:
            mean = st["pixel_sum"] / st["pixel_count"]
            var = (st["pixel_sq_sum"] / st["pixel_count"]) - mean**2
            var = max(var, 0)
            std = float(np.sqrt(var))
        else:
            mean = std = 0.0

        output[c] = {
            "num_images": st["num_images"],
            "shape_counts": st["shape_counts"],
            "pixel_stats": {
                "min": float(st["min"]) if st["min"] < float("inf") else None,
                "max": float(st["max"]) if st["max"] > float("-inf") else None,
                "mean": float(mean),
                "std": float(std),
            }
        }

    return output
