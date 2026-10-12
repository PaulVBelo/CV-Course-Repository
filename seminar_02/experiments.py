import gc
import json
from pathlib import Path

import pandas as pd
import torch

from .finetune import finetune_model
from .runtime import environment_info
from .train_linear_probe import train_linear_probe


def run_experiment(config: dict, backbone_name: str, mode: str,
                   probe_checkpoint: str | None = None, reuse: bool = True) -> dict:
    if mode not in ("linear_probe", "finetune"):
        raise ValueError("mode must be linear_probe or finetune")

    if reuse:
        environment = environment_info()
        candidates = sorted(Path(config["results_dir"]).glob(f"{backbone_name}_{mode}_*/metrics.json"), reverse=True)

        for path in candidates:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                continue

            saved_config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            saved_environment = json.loads((path.parent / "environment.json").read_text(encoding="utf-8"))
            if saved_config != config or saved_environment != environment:
                continue
            if mode == "finetune" and result.get("head_initialization") != (probe_checkpoint or "random"):
                continue
            if not Path(result["checkpoint"]).is_file():
                raise FileNotFoundError(f"Checkpoint missing: {result['checkpoint']}. Use reuse=False to retrain.")

            print(f"{backbone_name}: загружен {path.parent.name}")
            return result

    print(f"{backbone_name}: {mode}", flush=True)
    try:
        if mode == "linear_probe":
            return train_linear_probe(config, backbone_name)

        return finetune_model(config, backbone_name, probe_checkpoint)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def load_saved_experiment(config: dict, backbone_name: str, mode: str, manifest: dict,
                          probe_checkpoint: str | None = None) -> dict:
    if mode not in ("linear_probe", "finetune"):
        raise ValueError("mode must be linear_probe or finetune")

    candidates = sorted(Path(config["results_dir"]).glob(f"{backbone_name}_{mode}_*/metrics.json"), reverse=True)
    for path in candidates:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("manifest") != manifest:
            continue
        if result.get("backbone") != backbone_name or result.get("mode") != mode:
            continue

        saved_config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        keys = ("seed", "image_size", "amp", "amp_dtype", "linear_probe")
        if mode == "finetune":
            keys += ("finetune",)
            if saved_config["data"]["batch_size"] != config["data"]["batch_size"]:
                continue
        if any(saved_config.get(key) != config.get(key) for key in keys):
            continue

        if mode == "finetune" and probe_checkpoint is not None:
            initialization = result.get("head_initialization")
            if not initialization or Path(initialization).resolve() != Path(probe_checkpoint).resolve():
                continue

        result["output_dir"] = str(path.parent.resolve())
        print(f"{backbone_name}: загружен {path.parent.name}")
        return result

    raise FileNotFoundError(
        f"No completed {backbone_name}/{mode} result matches this configuration and split in "
        f"{config['results_dir']}. Training is disabled; restore the original configuration or results.")


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    protocols = {result["protocol"] for result in results.values()}
    if len(protocols) != 1:
        raise ValueError("Results must use the same data split")

    rows = []
    for name, result in results.items():
        history = pd.read_csv(Path(result["output_dir"]) / "history.csv")
        best = history.loc[history["epoch"] == result["best_epoch"]].iloc[0]
        rows.append({"backbone": name, "mode": result["mode"], "best_epoch": result["best_epoch"],
                     "val_macro_f1": result["best_val_macro_f1"],
                     **{f"test_{key}": result["test"][key] for key in ("loss", "accuracy", "macro_f1")},
                     "train_val_f1_gap": float(best["train_macro_f1"] - best["val_macro_f1"])})

    return pd.DataFrame(rows).set_index("backbone")


def compare_results(probes: dict[str, dict], finetuned: dict[str, dict], path=None) -> pd.DataFrame:
    if probes.keys() != finetuned.keys():
        raise ValueError("Both methods must contain the same backbones")

    protocols = {result["protocol"] for result in [*probes.values(), *finetuned.values()]}
    if len(protocols) != 1:
        raise ValueError("Results must use the same data split")

    linear = results_table(probes)
    tuned = results_table(finetuned)
    comparison = pd.DataFrame(index=linear.index)

    for metric in ("accuracy", "macro_f1"):
        comparison[f"LP {metric}"] = linear[f"test_{metric}"]
        comparison[f"FT {metric}"] = tuned[f"test_{metric}"]
        comparison[f"delta {metric}, pp"] = 100 * (tuned[f"test_{metric}"] - linear[f"test_{metric}"])

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(path)

    return comparison


def summarize_results(probes: dict[str, dict], finetuned: dict[str, dict]) -> str:
    comparison = compare_results(probes, finetuned)
    records = [*probes.values(), *finetuned.values()]
    selected = max(records, key=lambda result: result["best_val_macro_f1"])
    test = selected["test"]
    lines = [f"По validation macro-F1 выбрана {selected['backbone']} / {selected['mode']}: "
             f"{selected['best_val_macro_f1']:.3f}. На test: accuracy {test['accuracy']:.3f}, "
             f"macro-F1 {test['macro_f1']:.3f}."]

    for name, row in comparison.iterrows():
        delta = row["delta macro_f1, pp"]
        lines.append(f"{name}: изменение test macro-F1 после fine-tuning {delta:+.2f} п.п.")

    improvements = comparison["delta macro_f1, pp"]
    leader = improvements.idxmax()
    if improvements[leader] > 0:
        lines.append(f"Наибольший прирост macro-F1 у {leader}: {improvements[leader]:+.2f} п.п.")
    else:
        lines.append("Fine-tuning не повысил test macro-F1 ни у одного backbone.")

    matrix = test["confusion_matrix"]
    mistakes = [(matrix[i][j], i, j) for i in range(len(matrix)) for j in range(len(matrix)) if i != j]
    count, actual, predicted = max(mistakes)
    if count:
        names = selected["class_names"]
        lines.append(f"Самая частая ошибка выбранной модели: {names[actual]} → {names[predicted]}, "
                     f"{count} изображений.")

    return "\n\n".join(lines)
