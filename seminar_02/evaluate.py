from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F

from .runtime import autocast_context


def classification_metrics(targets, predictions, num_classes: int) -> dict:
    return {"accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(num_classes)),
                                       average="macro", zero_division=0))}


def evaluate_model(model, loader, device, config: dict, num_classes: int) -> dict:
    device = torch.device(device)
    model.eval()

    targets, predictions, probabilities = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast_context(device, config):
                logits = model(inputs)

            logits = logits.float()
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())

    metrics = classification_metrics(targets, predictions, num_classes)
    return {**metrics, "loss": total_loss / len(targets), "targets": targets,
            "predictions": predictions, "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(targets, predictions, labels=range(num_classes)).tolist()}


def error_analysis(result: dict, class_names: list[str]) -> pd.DataFrame:
    targets, predictions = np.array(result["targets"]), np.array(result["predictions"])
    return pd.DataFrame([
        {"class": name, "false_positives": int(((predictions == i) & (targets != i)).sum()),
         "false_negatives": int(((predictions != i) & (targets == i)).sum()),
         "support": int((targets == i).sum())} for i, name in enumerate(class_names)])


def comparison_table(results_dir: str | Path, protocol: str | None = None) -> pd.DataFrame:
    import json

    rows = []

    for name in ("supervised", "mae", "dinov2"):
        for mode in ("linear_probe", "finetune"):
            records = []
            for path in sorted(Path(results_dir).glob(f"{name}_{mode}_*/metrics.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] == "completed" and (protocol is None or record["protocol"] == protocol):
                    records.append((path, record))

            if not records:
                rows.append({"backbone": name, "mode": mode, "status": "not_run"})

            for path, record in records:
                rows.append({"backbone": name, "mode": mode, "status": "completed",
                             "protocol": record["protocol"], "run": path.parent.name,
                             **{key: record["test"][key] for key in ("loss", "accuracy", "macro_f1")}})

    return pd.DataFrame(rows)


def classification_reports(results: dict[str, dict]) -> pd.DataFrame:
    from sklearn.metrics import classification_report

    frames = {}
    for name, result in results.items():
        test = result["test"]
        options = {"labels": list(range(len(result["class_names"]))),
                   "target_names": result["class_names"], "zero_division": 0}
        report = classification_report(test["targets"], test["predictions"], output_dict=True, **options)
        frame = pd.DataFrame({label: values for label, values in report.items() if isinstance(values, dict)}).T
        frame["support"] = frame["support"].astype(int)
        frames[name] = frame

        output = Path(result["output_dir"])
        frame.to_csv(output / "classification_report.csv", index_label="class")
        (output / "classification_report.txt").write_text(
            classification_report(test["targets"], test["predictions"], digits=4, **options), encoding="utf-8")

    return pd.concat(frames, names=["backbone", "class"])
