import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "seminar_02"

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import create_dataloaders
from .engine import experiment_metadata, fit
from .features import cached_features
from .models import MODEL_NAMES, load_backbone
from .runtime import load_config, prepare_run, resolve_device, seed_everything


def train_linear_probe(config: dict, backbone_name: str = "dinov2", smoke: bool = False) -> dict:
    seed_everything(config["seed"])
    device = resolve_device(config["device"])

    bundle = create_dataloaders(config)
    backbone = load_backbone(backbone_name, device)
    fingerprint = backbone.fingerprint()
    datasets = cached_features(backbone, bundle, config)

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    settings = config["linear_probe"]
    loaders = {split: DataLoader(dataset, batch_size=settings["batch_size"], shuffle=split == "train",
                                 generator=torch.Generator().manual_seed(config["seed"]))
               for split, dataset in datasets.items()}

    head = nn.Linear(datasets["train"].tensors[0].shape[1], len(bundle.class_names)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])

    output, checkpoint = prepare_run(config, backbone_name, "linear_probe", smoke)
    metadata = experiment_metadata(backbone_name, "linear_probe", config, bundle, fingerprint, smoke)

    return fit(head, loaders, optimizer, config, settings, len(bundle.class_names), checkpoint, output, metadata)


def load_probe_head(checkpoint: str | Path, backbone_name: str, config: dict, bundle, fingerprint: str):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = experiment_metadata(backbone_name, "linear_probe", config, bundle, fingerprint, False)

    for key in ("backbone", "protocol", "weights_sha256", "class_names"):
        if saved["metadata"][key] != expected[key]:
            raise ValueError(f"Probe checkpoint mismatch: {key}")
    if saved["metadata"]["status"] != "completed":
        raise ValueError("A smoke checkpoint cannot initialize a full fine-tuning run")

    return saved["model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", choices=MODEL_NAMES, default="dinov2")
    args = parser.parse_args()

    result = train_linear_probe(load_config(args.config), args.backbone)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
