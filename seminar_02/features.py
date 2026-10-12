from pathlib import Path

import numpy as np
import timm
import torchvision
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from .runtime import autocast_context, digest


def extract_features(backbone, loader: DataLoader, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Feature extraction requires a frozen backbone")

    backbone.eval()
    device = next(backbone.parameters()).device
    features, labels = [], []

    with torch.no_grad():
        for images, target in loader:
            with autocast_context(device, config):
                encoded = backbone(images.to(device, non_blocking=True))

            if encoded.ndim != 2 or not torch.isfinite(encoded).all():
                raise RuntimeError("Invalid backbone features")

            features.append(encoded.float().cpu())
            labels.append(target.cpu())

    return torch.cat(features), torch.cat(labels)


def cached_features(backbone, bundle, config: dict,
                    splits: tuple[str, ...] | None = None) -> dict[str, TensorDataset]:
    fingerprint = backbone.fingerprint()
    result = {}

    for split in splits if splits is not None else bundle.loaders:
        loader = bundle.loaders[split]
        if loader.dataset.augmented or not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("Cache requires deterministic transforms and sequential sampling")

        metadata = {"schema": 1, "weights": fingerprint, "manifest": bundle.manifest,
                    "split": split, "amp": config["amp"], "dtype": config["amp_dtype"],
                    "device": str(next(backbone.parameters()).device), "torch": str(torch.__version__),
                    "timm": timm.__version__, "torchvision": torchvision.__version__}
        path = Path(config["cache_dir"]) / f"{backbone.name}_{split}_{digest(metadata)}.pt"

        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["metadata"] != metadata:
                raise RuntimeError("Feature cache metadata mismatch")
            features, labels = saved["features"], saved["labels"]
        else:
            print(f"{backbone.name}: extracting {split} features ({len(loader.dataset)} images)", flush=True)
            features, labels = extract_features(backbone, loader, config)

            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save({"metadata": metadata, "features": features, "labels": labels}, temporary)
            temporary.replace(path)

        result[split] = TensorDataset(features, labels)

    return result



def comparison_features(config: dict, bundle) -> tuple[dict, np.ndarray]:
    import gc
    from .models import MODEL_NAMES, load_backbone

    embeddings = {}
    reference_labels = None

    for name in MODEL_NAMES:
        backbone = load_backbone(name, config["device"])
        features, labels = cached_features(backbone, bundle, config, splits=("test",))["test"].tensors
        if reference_labels is not None and not torch.equal(labels, reference_labels):
            raise ValueError("Representations must use the same image order")

        embeddings[name] = features.numpy()
        reference_labels = labels
        del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return embeddings, reference_labels.numpy()
