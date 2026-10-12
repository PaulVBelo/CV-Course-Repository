import hashlib

import timm
import torch
from torch import nn

from .data import IMAGENET_MEAN, IMAGENET_STD

MODEL_NAMES = {
    "supervised": "vit_base_patch16_224.orig_in21k_ft_in1k",
    "mae": "vit_base_patch16_224.mae",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}


class Backbone(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.model_id = MODEL_NAMES[name]

        self.encoder = timm.create_model(self.model_id, pretrained=True, num_classes=0,
                                         img_size=224, global_pool="token")
        self.num_features = self.encoder.num_features

        mean = torch.tensor(self.encoder.pretrained_cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(self.encoder.pretrained_cfg["std"]).view(1, 3, 1, 1)
        self.register_buffer("input_scale", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) / std)
        self.register_buffer("input_offset", (torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) - mean) / std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images * self.input_scale + self.input_offset)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.model_id.encode())
        for name, value in self.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())

        return digest.hexdigest()


def load_backbone(name: str, device: str | torch.device = "cpu") -> Backbone:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown backbone: {name}; choose {list(MODEL_NAMES)}")

    return Backbone(name).requires_grad_(False).eval().to(device)


class ImageClassifier(nn.Module):
    def __init__(self, backbone: Backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, num_classes)

        self.unfrozen_blocks = 0
        self.fp32_head = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if self.fp32_head:
            with torch.autocast(device_type=images.device.type, enabled=False):
                return self.head(features.float())
        return self.head(features)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()

        if mode and self.unfrozen_blocks:
            for block in self.backbone.encoder.blocks[-self.unfrozen_blocks:]:
                block.train()
            self.backbone.encoder.norm.train()

        return self


def configure_partial_finetuning(model: ImageClassifier, blocks: int) -> None:
    if blocks not in (1, 2):
        raise ValueError("Partial fine-tuning supports the last 1 or 2 blocks")

    model.backbone.requires_grad_(False)
    for block in model.backbone.encoder.blocks[-blocks:]:
        block.requires_grad_(True)
    model.backbone.encoder.norm.requires_grad_(True)

    model.head.requires_grad_(True)
    model.unfrozen_blocks = blocks
