import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "seminar_02"

import argparse
import math

import torch

from .data import create_dataloaders
from .engine import experiment_metadata, fit, layerwise_parameter_groups, warmup_cosine
from .models import MODEL_NAMES, ImageClassifier, configure_partial_finetuning, load_backbone
from .runtime import load_config, prepare_run, resolve_device, save_json, seed_everything
from .train_linear_probe import load_probe_head


def finetune_model(config: dict, backbone_name: str = "dinov2", probe_checkpoint: str | None = None,
                   smoke: bool = False) -> dict:
    seed_everything(config["seed"])
    device = resolve_device(config["device"])

    bundle = create_dataloaders(config, augment_train=True)
    backbone = load_backbone(backbone_name, device)
    fingerprint = backbone.fingerprint()
    model = ImageClassifier(backbone, len(bundle.class_names)).to(device)
    if probe_checkpoint:
        model.head.load_state_dict(load_probe_head(probe_checkpoint, backbone_name, config, bundle, fingerprint))

    settings = config["finetune"]
    if not 0 < settings["layer_decay"] <= 1:
        raise ValueError("layer_decay must be in (0, 1]")

    configure_partial_finetuning(model, settings["unfreeze_blocks"])
    model.backbone.encoder.set_grad_checkpointing(settings["gradient_checkpointing"])

    groups = layerwise_parameter_groups(model, settings)
    optimizer = torch.optim.AdamW(groups)

    updates_per_epoch = math.ceil(len(bundle.loaders["train"]) / settings["accumulation_steps"])
    scheduler = warmup_cosine(optimizer, settings["epochs"] * updates_per_epoch,
                              settings["warmup_epochs"] * updates_per_epoch)

    output, checkpoint = prepare_run(config, backbone_name, "finetune", smoke)
    save_json(output / "parameter_groups.json", {
        "groups": [{key: value for key, value in group.items() if key != "params"} for group in groups],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    metadata = experiment_metadata(backbone_name, "finetune", config, bundle, fingerprint, smoke)
    metadata["head_initialization"] = probe_checkpoint or "random"

    return fit(model, bundle.loaders, optimizer, config, settings, len(bundle.class_names),
               checkpoint, output, metadata, scheduler)


def load_classifier(checkpoint: str, device: str = "cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]

    backbone = load_backbone(metadata["backbone"], device)
    model = ImageClassifier(backbone, len(metadata["class_names"])).to(device)

    if metadata["mode"] == "linear_probe":
        if backbone.fingerprint() != metadata["weights_sha256"]:
            raise ValueError("Pretrained weights differ from the probe checkpoint")
        model.head.load_state_dict(saved["model"])
        model.fp32_head = True
    else:
        model.load_state_dict(saved["model"])

    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", choices=MODEL_NAMES, default="dinov2")
    parser.add_argument("--probe-checkpoint", default=None)
    args = parser.parse_args()

    result = finetune_model(load_config(args.config), args.backbone, args.probe_checkpoint)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
