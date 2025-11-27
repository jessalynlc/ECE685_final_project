import pandas as pd
import numpy as np

def create_class_mappings(csv_file: str, label_column: str = "Finding Labels"):
    """
    Create class-index and index-class dictionaries from a CSV file where 
    labels can be multi-label strings separated by '|'.

    Args:
        csv_file (str): Path to the CSV file.
        label_column (str, optional): Column containing class labels. Defaults to "Finding Labels".

    Returns:
        tuple[dict, dict]: (class_to_idx, idx_to_class)
    """
    # Load CSV
    df = pd.read_csv(csv_file)

    # Split multi-labels by '|' and flatten to get all unique labels
    all_labels = df[label_column].dropna().apply(lambda x: x.split('|'))
    flat_labels = [label for sublist in all_labels for label in sublist]

    unique_labels = sorted(set(flat_labels))  # sort for deterministic indices

    # Create mappings
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class


import pandas as pd
import numpy as np

def create_image_multihot_mapping_from_dicts(csv_file: str, class_to_idx: dict, label_column: str = "Finding Labels", image_column: str = "Image Index"):
    """
    Create a dictionary mapping image paths to multi-hot label vectors using
    existing class_to_idx mapping.

    Args:
        csv_file (str): Path to CSV file.
        class_to_idx (dict): Mapping from class label to index.
        label_column (str, optional): Column containing multi-label strings separated by '|'. Defaults to "Finding Labels".
        image_column (str, optional): Column containing image paths. Defaults to "Image Index".

    Returns:
        dict[str, np.ndarray]: Mapping image path → multi-hot vector.
    """
    df = pd.read_csv(csv_file)
    num_classes = len(class_to_idx)
    image_to_multihot = {}

    for _, row in df.iterrows():
        labels = str(row[label_column]).split("|") if pd.notna(row[label_column]) else []
        multihot = np.zeros(num_classes, dtype=np.float32)
        for label in labels:
            if label in class_to_idx:
                multihot[class_to_idx[label]] = 1.0
        image_to_multihot[row[image_column]] = multihot

    return image_to_multihot
