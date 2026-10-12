import math
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .evaluate import classification_metrics, evaluate_model
from .runtime import autocast_context, digest, save_json


def layerwise_parameter_groups(model, settings: dict) -> list[dict]:
    groups = {}
    depth = len(model.backbone.encoder.blocks)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("head."):
            lr = settings["head_lr"]
        elif ".blocks." in name:
            layer = int(name.split(".blocks.")[1].split(".")[0])
            lr = settings["lr"] * settings["layer_decay"] ** (depth - 1 - layer)
        else:
            lr = settings["lr"]

        decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else settings["weight_decay"]
        key = (lr, decay)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay, "names": []})
        groups[key]["params"].append(parameter)
        groups[key]["names"].append(name)

    return list(groups.values())


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def fit(model, loaders: dict, optimizer, config: dict, settings: dict,
        num_classes: int, checkpoint: Path, output: Path, metadata: dict,
        scheduler=None) -> dict:
    device = next(model.parameters()).device
    config = dict(config)
    if metadata["mode"] == "linear_probe":
        config["amp"] = False

    epochs = settings["epochs"]
    accumulation = settings["accumulation_steps"]
    if epochs < 1 or accumulation < 1:
        raise ValueError("epochs and accumulation_steps must be positive")

    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=config["amp"] and device.type == "cuda"
                                 and config["amp_dtype"] == "float16")

    history, best_score, updates = [], -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        targets, predictions, loss_sum, window_samples = [], [], 0.0, 0
        lr_used = max(group["lr"] for group in optimizer.param_groups)

        for index, (inputs, labels) in enumerate(loaders["train"]):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast_context(device, config):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels, reduction="sum")

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            scaler.scale(loss).backward()
            window_samples += len(labels)

            if (index + 1) % accumulation == 0 or index + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)

                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("clip_grad_norm", float("inf")))

                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= previous_scale:
                    updates += 1
                    if scheduler is not None:
                        scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                window_samples = 0

            loss_sum += loss.detach().float().item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())

        train = classification_metrics(targets, predictions, num_classes)
        validation = evaluate_model(model, loaders["val"], device, config, num_classes)

        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(targets),
               **{f"train_{key}": value for key, value in train.items()},
               **{f"val_{key}": validation[key] for key in ("loss", "accuracy", "macro_f1")},
               "max_lr": lr_used, "optimizer_updates": updates}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)

        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "metadata": metadata, "epoch": epoch + 1, "val_macro_f1": best_score}, checkpoint)

        if (epoch + 1) % max(1, config.get("log_every", 1)) == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1} train_loss={row['train_loss']:.4f} "
                  f"val_f1={validation['macro_f1']:.4f}", flush=True)

    if updates == 0:
        raise RuntimeError("No successful optimizer update")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    test = evaluate_model(model, loaders["test"], device, config, num_classes)

    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    result = {**metadata, "best_epoch": saved["epoch"], "best_val_macro_f1": best_score,
              "test": test, "peak_cuda_memory_mib": peak, "checkpoint": str(checkpoint),
              "output_dir": str(output)}

    save_json(output / "metrics.json", result)
    pd.DataFrame({"target": test["targets"], "prediction": test["predictions"]}).to_csv(
        output / "predictions.csv", index=False)

    from .visualize import plot_confusion_matrix, plot_history
    from .evaluate import error_analysis
    import matplotlib.pyplot as plt

    plt.close(plot_confusion_matrix(test, metadata["class_names"], output / "confusion_matrix.png"))
    plt.close(plot_history(output / "history.csv", output / "history.png"))
    error_analysis(test, metadata["class_names"]).to_csv(output / "errors.csv", index=False)

    return result


def experiment_metadata(name: str, mode: str, config: dict, bundle, fingerprint: str, smoke: bool) -> dict:
    return {"backbone": name, "mode": mode, "status": "smoke_only" if smoke else "completed",
            "protocol": digest({"manifest": bundle.manifest, "seed": config["seed"]}),
            "class_names": bundle.class_names, "weights_sha256": fingerprint,
            "manifest": bundle.manifest}
