# ECE685_final_project

The file hierarchy in archive should look like this:
{
  "_files": [
    "BBox_List_2017_Official_NIH.csv",
    "test_list_NIH.txt",
    "pretrained_model.h5",
    "train_val_list_NIH.txt",
    "Data_Entry_2017.csv"
  ],
  "images-224": {
    "images-224": "..."
  }
}
Otherwise change the links in setup

This repository contains a full training / validation / testing pipeline for multi-label medical image classification.
The main workflow is implemented in main.ipynb, which orchestrates:

- dataset loading
- multi-hot label creation
- transforms & augmentations
- training & validation loops
- checkpointing
- testing / prediction

Model architectures and training logic are modularized under the models/ directory, and generic utilities live under helpers_functions/.

Repository Structure:
.
├── main.ipynb                 # Main notebook: runs training, validation, testing
│
├── helpers_functions/         # General utility scripts
│   ├── data_stats.py          # Dataset statistics
│   ├── metrics.py             # Evaluation metrics (AUC, F1, thresholds, etc.)
│   ├── multi_hot.py           # Multi-hot label encoding
│   └── setup.py               # Misc setup helpers
│
├── models/                    # All model architectures and training logic
│   ├── models.py              # Central dispatcher: selects which model to build
│   ├── train.py               # General training wrapper
│   ├── val.py                 # General validation wrapper
│   │
│   ├── manifold_mixup/        # ManiFold Mixup model
│   │   ├── resnet_manifold_mixup.py
│   │   ├── train.py
│   │   ├── val.py
│   │   └── pred.py
│   │
│   ├── mo_ex/                 # Mixture of Experts model
│   │   ├── resnet_mo_ex.py
│   │   ├── mo_ex_source.py
│   │   ├── train.py
│   │   ├── val.py
│   │   └── pred.py
│   │
│   ├── aug_all/               # Augmentation-heavy model variants
│   │   ├── resnet_aug_all.py
│   │   ├── aug_all.py
│   │   ├── train.py
│   │   ├── val.py
│   │   └── pred.py
│   │
│   └── asl/                   # ASL model variant
│       ├── aug_all.py
│       ├── train.py
│       ├── val.py
│       └── pred.py
│
├── checkpoints/               # Saved model checkpoints
├── results/                   # Output metrics / predictions and threshold results
└── README.md
