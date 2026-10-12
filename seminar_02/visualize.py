from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


def save_figure(figure, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")

    return figure


def image_array(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


def plot_samples(dataset, class_names: list[str], count: int = 9, path=None,
                 one_per_class: bool = False):
    if count < 1 or not len(dataset):
        raise ValueError("Use a nonempty dataset and a positive count")

    if one_per_class:
        labels = np.asarray(dataset.source.targets)[dataset.indices]
        indices = [int(np.flatnonzero(labels == label)[0]) for label in dataset.classes][:count]
    else:
        indices = list(range(min(count, len(dataset))))

    columns = min(5, len(indices))
    rows = (len(indices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.5 * rows), squeeze=False)

    for axis in axes.flat:
        axis.axis("off")

    for index, axis in zip(indices, axes.flat):
        image, label = dataset[index]
        axis.imshow(image_array(image))
        axis.set_title(class_names[label].replace("_", " "), fontsize=10)

    figure.tight_layout()
    return save_figure(figure, path)


def plot_confusion_matrix(result: dict, class_names: list[str], path=None):
    import seaborn as sns

    size = max(6, 0.5 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size - 1))
    sns.heatmap(np.array(result["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 7}, ax=axis)
    axis.set(xlabel="Predicted", ylabel="True")
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    plt.setp(axis.get_xticklabels(), ha="right", va="top", rotation_mode="anchor")
    axis.tick_params(axis="y", labelrotation=0, labelsize=8)
    figure.tight_layout()

    return save_figure(figure, path)


def plot_history(history, path=None):
    import pandas as pd

    frame = pd.read_csv(history) if isinstance(history, (str, Path)) else history
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for axis, metric in zip(axes, ("loss", "accuracy", "macro_f1")):
        for split in ("train", "val"):
            axis.plot(frame["epoch"], frame[f"{split}_{metric}"], label=split)
        axis.set(xlabel="Epoch", ylabel=metric)
        axis.legend()

    return save_figure(figure, path)


def compare_embeddings(embeddings: dict, labels, class_names: list[str], method: str = "tsne",
                       seed: int = 42, max_samples: int = 1000, path=None):
    labels = np.asarray(labels)
    if len(labels) < 5:
        raise ValueError("Embedding plots require at least five samples")

    indices = np.random.default_rng(seed).permutation(len(labels))[:max_samples]
    if len(indices) < 5:
        raise ValueError("max_samples must be at least five")

    legend_rows = (len(class_names) + 4) // 5
    figure, axes = plt.subplots(1, len(embeddings), figsize=(5 * len(embeddings), 4 + 0.3 * legend_rows), squeeze=False)
    palette = plt.get_cmap("tab20", len(class_names))
    coordinates = {}

    for axis, (name, features) in zip(axes.flat, embeddings.items()):
        features = np.asarray(features)
        if len(features) != len(labels):
            raise ValueError("All representations must use the same images in the same order")

        values = features[indices].astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        values = PCA(n_components=min(50, len(indices) - 1, values.shape[1]), random_state=seed).fit_transform(values)

        if method == "tsne":
            projection = TSNE(n_components=2, perplexity=min(30, (len(indices) - 1) / 3),
                              init="pca", learning_rate="auto", random_state=seed).fit_transform(values)
        elif method == "umap":
            from umap import UMAP
            projection = UMAP(n_components=2, n_neighbors=min(15, len(indices) - 1),
                              random_state=seed, n_jobs=1).fit_transform(values)
        else:
            raise ValueError("method must be tsne or umap")

        coordinates[name] = projection
        for label, class_name in enumerate(class_names):
            selected = labels[indices] == label
            axis.scatter(*projection[selected].T, s=15, label=class_name.replace("_", " "),
                         color=palette(label), alpha=0.75)
        axis.set_title(f"{name}: {method}")
        axis.set_xticks([])
        axis.set_yticks([])

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=min(5, len(class_names)), frameon=False)
    figure.tight_layout(rect=(0, 0.06 * legend_rows + 0.03, 1, 1))
    return save_figure(figure, path), coordinates


def attention_rollout(backbone, image: torch.Tensor) -> torch.Tensor:
    if backbone.name != "dinov2":
        raise ValueError("This attention rollout is validated for the timm DINOv2 backbone")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("Use one image per attention visualization")

    modes = {module: module.training for module in backbone.modules()}
    fused, hooks, matrices = [], [], []

    def capture(module, inputs, output):
        if output.ndim != 4:
            raise RuntimeError("Expected genuine attention probabilities [B, heads, tokens, tokens]")
        matrices.append(output.detach().float().mean(1).cpu())

    try:
        backbone.eval()
        for block in backbone.encoder.blocks:
            attention = block.attn
            fused.append((attention, attention.fused_attn))
            attention.fused_attn = False
            hooks.append(attention.attn_drop.register_forward_hook(capture))

        with torch.no_grad():
            backbone(image.to(next(backbone.parameters()).device).float())
        if len(matrices) != len(backbone.encoder.blocks):
            raise RuntimeError("Not all attention layers were captured")

        tokens = matrices[0].shape[-1]
        identity = torch.eye(tokens).unsqueeze(0)
        rollout = identity.clone()
        for attention in matrices:
            attention = attention + identity
            attention = attention / attention.sum(-1, keepdim=True)
            rollout = attention @ rollout

        prefix = backbone.encoder.num_prefix_tokens
        grid = backbone.encoder.patch_embed.grid_size
        mask = rollout[0, 0, prefix:].reshape(1, 1, *grid)
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return (mask - mask.min()) / (mask.max() - mask.min()).clamp_min(1e-12)
    finally:
        for hook in hooks:
            hook.remove()

        for attention, original in fused:
            attention.fused_attn = original

        for module, mode in modes.items():
            module.training = mode


def plot_attention(backbone, image: torch.Tensor, path=None):
    mask = attention_rollout(backbone, image)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis in axes:
        axis.imshow(image_array(image))
        axis.axis("off")

    axes[1].imshow(mask.numpy(), cmap="inferno", alpha=0.5)
    axes[0].set_title("Input crop")
    axes[1].set_title("DINOv2 attention rollout")

    return save_figure(figure, path)


def plot_errors(dataset, result: dict, class_names: list[str], class_index: int,
                kind: str = "fp", count: int = 6, path=None):
    if getattr(dataset, "augmented", False):
        raise ValueError("Use the deterministic test dataset")

    truth, prediction = np.asarray(result["targets"]), np.asarray(result["predictions"])
    if len(truth) != len(dataset):
        raise ValueError("Dataset must match the evaluation order and size")
    if kind not in ("fp", "fn"):
        raise ValueError("kind must be fp or fn")

    mask = ((prediction == class_index) & (truth != class_index)) if kind == "fp" else (
        (truth == class_index) & (prediction != class_index))
    indices = np.flatnonzero(mask)
    confidence = np.asarray(result["probabilities"]).max(1)
    indices = indices[np.argsort(-confidence[indices])][:count]

    figure, axes = plt.subplots(1, max(1, len(indices)), figsize=(3 * max(1, len(indices)), 3), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    if not len(indices):
        axes[0, 0].text(0.1, 0.5, f"No {kind} for {class_names[class_index]}")

    for axis, index in zip(axes.flat, indices):
        image, label = dataset[int(index)]
        if label != truth[index]:
            raise ValueError("Dataset labels do not match evaluation order")
        axis.imshow(image_array(image))
        axis.set_title(f"True: {class_names[label]}\nPred: {class_names[prediction[index]]}\n"
                       f"Confidence: {confidence[index]:.1%}", fontsize=10)

    figure.suptitle(f"{class_names[class_index]} · {kind.upper()}")
    figure.tight_layout()
    return save_figure(figure, path)


def plot_learning_curves(results: dict[str, dict], path=None):
    import pandas as pd

    figure, axes = plt.subplots(len(results), 3, figsize=(12, 2.8 * len(results)), squeeze=False)
    for row, (name, result) in enumerate(results.items()):
        history = pd.read_csv(Path(result["output_dir"]) / "history.csv")
        for axis, metric in zip(axes[row], ("loss", "accuracy", "macro_f1")):
            for split in ("train", "val"):
                axis.plot(history["epoch"], history[f"{split}_{metric}"], label=split, marker=".", markersize=4)
            axis.axvline(result["best_epoch"], color="0.5", linestyle=":", linewidth=1)
            axis.set(title=f"{name} · {metric}", xlabel="Epoch")
            axis.legend(frameon=False)

    figure.tight_layout()
    return save_figure(figure, path)


def plot_confusion_matrices(probes: dict[str, dict], finetuned: dict[str, dict], path=None):
    import seaborn as sns

    if probes.keys() != finetuned.keys():
        raise ValueError("Both methods must contain the same backbones")

    maximum = max(np.asarray(result["test"]["confusion_matrix"]).max()
                  for result in [*probes.values(), *finetuned.values()])
    size = max(4, 0.35 * len(next(iter(probes.values()))["class_names"]))
    figure, axes = plt.subplots(len(probes), 2, figsize=(2 * size, size * len(probes)), squeeze=False)

    for row, name in enumerate(probes):
        for column, (mode, results) in enumerate((("Linear probe", probes), ("Fine-tuning", finetuned))):
            axis = axes[row, column]
            result = results[name]
            sns.heatmap(np.asarray(result["test"]["confusion_matrix"]), annot=True, fmt="d",
                        cmap="Blues", cbar=False, vmin=0, vmax=maximum, xticklabels=result["class_names"],
                        yticklabels=result["class_names"], annot_kws={"fontsize": 8}, ax=axis)
            axis.set(title=f"{name} · {mode}", xlabel="Predicted", ylabel="True")
            axis.tick_params(axis="x", labelrotation=45, labelsize=10)
            plt.setp(axis.get_xticklabels(), ha="right", va="top", rotation_mode="anchor")
            axis.tick_params(axis="y", labelrotation=0, labelsize=10)

    figure.tight_layout()
    return save_figure(figure, path)


def plot_metric_comparison(comparison, path=None):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(comparison))

    for axis, metric in zip(axes, ("accuracy", "macro_f1")):
        axis.bar(positions - 0.18, comparison[f"LP {metric}"], width=0.36, label="Linear probe")
        axis.bar(positions + 0.18, comparison[f"FT {metric}"], width=0.36, label="Fine-tuning")
        axis.set(xticks=positions, xticklabels=comparison.index, ylim=(0, 1), ylabel=metric, title=f"Test {metric}")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return save_figure(figure, path)


def plot_attention_examples(config: dict, dataset, class_names: list[str], indices=(0, 1), path=None):
    import gc
    from .models import load_backbone

    backbone = load_backbone("dinov2", config["device"])
    figure, axes = plt.subplots(len(indices), 2, figsize=(7, 3.2 * len(indices)), squeeze=False)

    try:
        for row, index in enumerate(indices):
            image, label = dataset[index]
            mask = attention_rollout(backbone, image)
            for axis in axes[row]:
                axis.imshow(image_array(image))
                axis.axis("off")
            axes[row, 0].set_title(class_names[label])
            axes[row, 1].imshow(mask.numpy(), cmap="inferno", alpha=0.5)
            axes[row, 1].set_title("Pretrained DINOv2 · attention rollout")

        figure.tight_layout()
        return save_figure(figure, path)
    finally:
        del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
