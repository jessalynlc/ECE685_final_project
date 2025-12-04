import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_recall_fscore_support

device = "cuda" if torch.cuda.is_available() else "cpu"
path_checkpoint = "checkpoints/ManiFold_MixUp/ManiFold_Mixup_epoch005-best.pth"  # adjust if needed
save_dir = "thresholds"
os.makedirs(save_dir, exist_ok=True)

# Data: reuse existing val_dataset (random_split earlier)

dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Common patterns
    if isinstance(ckpt, dict):
        # Try common keys
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            # treat ckpt itself as state dict (some people save state_dict directly to file)
            state = ckpt
    else:
        state = ckpt

    # If state dict keys are prefixed (like 'module.'), try to strip
    new_state = {}
    for k, v in state.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model

# PREDICT PROBABILITIES ON VAL SET
@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    all_scores = []
    all_targets = []
    filenames = []
    for imgs, targets, names in tqdm(loader, desc="Predict val"):
        imgs = imgs.to(device)
        outputs = model(imgs)          # logits
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_scores.append(probs)
        all_targets.append(targets.numpy())
        filenames.extend(names)
    all_scores = np.vstack(all_scores)
    all_targets = np.vstack(all_targets)
    return all_scores, all_targets, filenames

# THRESHOLD SEARCH
def find_best_thresholds_by_f1(y_true, y_score, steps=101):
    # y_true: (N, C) binary
    # y_score: (N, C) in [0,1]
    thresholds = np.linspace(0.0, 1.0, steps)
    C = y_true.shape[1]
    best_thresh_per_class = np.zeros(C)
    best_f1_per_class = np.zeros(C)

    for c in range(C):
        truths = y_true[:, c]
        scores = y_score[:, c]
        # if class has no positives in val, skip and use 0.5 default
        if truths.sum() == 0:
            best_thresh_per_class[c] = 0.5
            best_f1_per_class[c] = 0.0
            continue

        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh_per_class[c] = best_t
        best_f1_per_class[c] = best_f1

    return best_thresh_per_class, best_f1_per_class

def eval_with_thresholds(y_true, y_score, thresholds):
    # thresholds: either single float or array-like length C
    if np.isscalar(thresholds):
        preds = (y_score >= thresholds).astype(int)
    else:
        thresh = np.array(thresholds)[None, :]   # (1, C)
        preds = (y_score >= thresh).astype(int)

    # Per-class F1
    per_class_f1 = []
    C = y_true.shape[1]
    for c in range(C):
        per_class_f1.append(f1_score(y_true[:, c], preds[:, c], zero_division=0))
    per_class_f1 = np.array(per_class_f1)

    # Micro F1
    micro_f1 = f1_score(y_true.ravel(), preds.ravel(), zero_division=0)

    # AP and AUROC per class
    ap = []
    auroc = []
    for c in range(C):
        truths = y_true[:, c]
        scores = y_score[:, c]
        try:
            ap.append(average_precision_score(truths, scores))
        except Exception:
            ap.append(np.nan)
        try:
            auroc.append(roc_auc_score(truths, scores))
        except Exception:
            auroc.append(np.nan)
    ap = np.array(ap)
    auroc = np.array(auroc)

    return {
        "per_class_f1": per_class_f1,
        "micro_f1": micro_f1,
        "mean_ap": np.nanmean(ap),
        "mean_auroc": np.nanmean(auroc),
        "ap_per_class": ap,
        "auroc_per_class": auroc,
        "preds": preds
    }

def main_tune(model, ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    model = load_checkpoint_to_model(model, ckpt_path, device)

    y_score, y_true, filenames = get_probs_and_labels(model, dataloader_val, device)
    print("Val probs shape:", y_score.shape, "Val truth shape:", y_true.shape)

    # Find per-class thresholds
    best_th, best_f1 = find_best_thresholds_by_f1(y_true, y_score, steps=101)
    print("Best per-class thresholds:", best_th)
    print("Best per-class F1s:", best_f1)

    # Also find a single global threshold maximizing micro F1
    thresholds_grid = np.linspace(0.0, 1.0, 101)
    best_micro = -1.0
    best_global_t = 0.5
    for t in thresholds_grid:
        res = eval_with_thresholds(y_true, y_score, t)
        if res["micro_f1"] > best_micro:
            best_micro = res["micro_f1"]
            best_global_t = t
    print(f"Best global threshold (micro-F1): {best_global_t:.2f}  micro-F1={best_micro:.4f}")

    # Evaluate using per-class thresholds
    res_perclass = eval_with_thresholds(y_true, y_score, best_th)
    print(f"Per-class thresholds -> mean AP: {res_perclass['mean_ap']:.4f}, mean AUROC: {res_perclass['mean_auroc']:.4f}, microF1: {res_perclass['micro_f1']:.4f}")

    # Evaluate using global threshold
    res_global = eval_with_thresholds(y_true, y_score, best_global_t)
    print(f"Global threshold {best_global_t:.2f} -> mean AP: {res_global['mean_ap']:.4f}, mean AUROC: {res_global['mean_auroc']:.4f}, microF1: {res_global['micro_f1']:.4f}")

    # Save thresholds and results
    out = {
        "best_per_class_thresholds": best_th.tolist(),
        "best_per_class_f1s": best_f1.tolist(),
        "best_global_threshold": float(best_global_t),
        "best_global_micro_f1": float(best_micro),
        "per_class_mean_ap": res_perclass["ap_per_class"].tolist(),
        "per_class_mean_auroc": res_perclass["auroc_per_class"].tolist()
    }
    out_path = os.path.join(save_dir, "thresholds_ManifoldMixup.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved thresholds to:", out_path)

    return out, y_score, y_true, filenames

if __name__ == "__main__":
    # must have `model` and `val_dataset` in scope, like from your notebook.
    out, scores, truths, fnames = main_tune(model, path_checkpoint)
    print(json.dumps(out, indent=2))