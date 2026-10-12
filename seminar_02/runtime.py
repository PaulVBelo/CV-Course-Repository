import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "seminar_02/config.yaml"
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)

    if config["image_size"] != 224:
        raise ValueError("This protocol uses 224px for all three backbones")

    for key in ("cache_dir", "results_dir", "checkpoint_dir"):
        config[key] = str((ROOT / config[key]).resolve())
    config["data"]["root"] = str((ROOT / config["data"]["root"]).resolve())

    return config


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly set device: cpu if desired")
    return device


def autocast_context(device: torch.device, config: dict):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config["amp_dtype"]]
    return torch.autocast(device.type, dtype=dtype, enabled=config["amp"] and device.type == "cuda")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def environment_info() -> dict:
    import platform
    import timm
    import torchvision

    _, total_mem = torch.cuda.mem_get_info()

    return {
        "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram": f"{total_mem // (1024**2)} MB" if torch.cuda.is_available() else None,
        }


def prepare_run(config: dict, name: str, mode: str, smoke: bool) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{name}_{mode}_{'smoke_' if smoke else ''}{stamp}"

    output = Path(config["results_dir"]) / run_id
    checkpoint = Path(config["checkpoint_dir"]) / run_id / "best.pt"
    output.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)

    save_json(output / "config.json", config)
    save_json(output / "environment.json", environment_info())

    return output, checkpoint
