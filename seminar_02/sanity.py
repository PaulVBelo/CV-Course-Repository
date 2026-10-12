import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "seminar_02"

import argparse
import copy
import gc

import torch
from torch import nn
from torch.nn import functional as F

from .data import create_dataloaders
from .evaluate import comparison_table
from .features import cached_features
from .finetune import finetune_model
from .models import MODEL_NAMES, ImageClassifier, configure_partial_finetuning, load_backbone
from .runtime import autocast_context, environment_info, load_config, resolve_device, save_json, seed_everything
from .train_linear_probe import train_linear_probe
from .visualize import compare_embeddings, plot_attention, plot_confusion_matrix, plot_history


def overfit_head(features: torch.Tensor, labels: torch.Tensor, num_classes: int,
                 steps: int = 150, lr: float = 0.01) -> dict:
    head = nn.Linear(features.shape[1], num_classes).to(features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    losses = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features), labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features)
        accuracy = (logits.argmax(1) == labels).float().mean().item()
        final_loss = F.cross_entropy(logits, labels).item()

    return {"initial_loss": losses[0], "final_loss": final_loss, "accuracy": accuracy,
            "steps": steps, "passed": accuracy >= 0.95 and final_loss < losses[0]}


def smoke_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["data"].update(train_per_class=6, val_from_train_per_class=2, val_from_test_per_class=1,
                          test_per_class=3, batch_size=2, num_workers=0)
    config["linear_probe"].update(epochs=1, batch_size=8)
    config["finetune"].update(epochs=1, accumulation_steps=4, warmup_epochs=0)
    return config


def run_smoke_tests(config: dict, names=None) -> dict:
    config = smoke_config(config)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    bundle = create_dataloaders(config)

    output = Path(config["results_dir"]) / "smoke"
    output.mkdir(parents=True, exist_ok=True)

    report = {"environment": environment_info(), "status": "smoke_only", "models": {}}
    embeddings = {}

    for name in names or MODEL_NAMES:
        print(f"Checking pretrained {name}", flush=True)
        backbone = load_backbone(name, device)
        features = cached_features(backbone, bundle, config)
        values, labels = features["train"].tensors
        assert values.shape == (6 * len(bundle.class_names), 768)

        tiny_values, tiny_labels = values[:4].to(device), labels[:4].to(device)
        overfit = overfit_head(tiny_values, tiny_labels, len(bundle.class_names))
        assert overfit["passed"], overfit

        model = ImageClassifier(backbone, len(bundle.class_names)).to(device)
        configure_partial_finetuning(model, 2)
        model.train()

        images, target = next(iter(bundle.loaders["train"]))
        with autocast_context(device, config):
            loss = F.cross_entropy(model(images.to(device)), target.to(device))
        loss.backward()

        active_gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in active_gradients)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert any(g.abs().sum() > 0 for g in active_gradients)

        embeddings[name] = features["test"].tensors[0].numpy()
        if name == "dinov2":
            plot_attention(backbone, bundle.loaders["test"].dataset[0][0], output / "attention_dinov2.png")

        report["models"][name] = {"shape": list(values.shape), "overfit": overfit,
                                  "finite_gradients": True, "frozen_gradients_absent": True}

        del active_gradients, model, backbone, features, loss
        gc.collect()
        torch.cuda.empty_cache()

        probe = train_linear_probe(config, name, smoke=True)
        tuned = finetune_model(config, name, smoke=True)

        for mode, result in (("linear_probe", probe), ("finetune", tuned)):
            plot_confusion_matrix(result["test"], bundle.class_names, output / f"{name}_{mode}_cm.png")
            plot_history(Path(result["output_dir"]) / "history.csv", output / f"{name}_{mode}_history.png")
            report["models"][name][mode] = {key: result[key] for key in (
                "test", "output_dir", "peak_cuda_memory_mib")}

        import matplotlib.pyplot as plt

        plt.close("all")
        save_json(output / "report.json", report)

    if embeddings:
        test_labels = [bundle.loaders["test"].dataset[i][1] for i in range(len(bundle.loaders["test"].dataset))]
        for method in ("tsne", "umap"):
            compare_embeddings(embeddings, test_labels, bundle.class_names, method=method,
                               path=output / f"{method}.png")

    comparison_table(config["results_dir"]).to_csv(Path(config["results_dir"]) / "comparison.csv", index=False)
    report["completed"] = True
    save_json(output / "report.json", report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backbone", choices=MODEL_NAMES, nargs="+")
    args = parser.parse_args()

    run_smoke_tests(load_config(args.config), args.backbone)


def backbone_checks(config: dict, bundle):
    import pandas as pd

    rows = []
    images, labels = next(iter(bundle.loaders["train"]))
    device = resolve_device(config["device"])

    for name in MODEL_NAMES:
        seed_everything(config["seed"])
        backbone = load_backbone(name, device)

        with torch.no_grad(), autocast_context(device, config):
            features = backbone(images.to(device)).float()

        result = overfit_head(features, labels.to(device), len(bundle.class_names))
        if features.shape != (len(labels), backbone.num_features) or not result["passed"]:
            raise RuntimeError(f"Backbone check failed: {name}, {result}")

        rows.append({"backbone": name, "feature_shape": str(tuple(features.shape)),
                     "initial_loss": result["initial_loss"], "final_loss": result["final_loss"],
                     "batch_accuracy": result["accuracy"]})
        del backbone, features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_everything(config["seed"])
    return pd.DataFrame(rows).set_index("backbone")


if __name__ == "__main__":
    main()
