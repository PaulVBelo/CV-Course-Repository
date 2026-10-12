from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FoodSource(ConcatDataset):
    def __init__(self, root: str):
        train = datasets.Food101(root, split="train", download=True)
        test = datasets.Food101(root, split="test", download=True)
        if train.classes != test.classes:
            raise ValueError("Official Food-101 splits must have matching classes")

        super().__init__([train, test])
        self.classes = train.classes
        self.train_size = len(train)
        self.targets = train._labels + test._labels
        self.image_ids = [path.relative_to(source._images_folder).as_posix()
                          for source in (train, test) for path in source._image_files]

        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("Official Food-101 metadata contains repeated image paths")


class ImageSubset(Dataset):
    def __init__(self, source, indices: list[int], classes: list[int], transform, augmented: bool):
        self.source = source
        self.indices = indices
        self.classes = classes
        self.transform = transform
        self.augmented = augmented
        self.mapping = {label: index for index, label in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.source[self.indices[index]]
        return self.transform(image), self.mapping[label]


@dataclass
class DataBundle:
    loaders: dict[str, DataLoader]
    class_names: list[str]
    manifest: dict


def seed_worker(worker_id: int) -> None:
    import random

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(config: dict, augment_train: bool = False) -> DataBundle:
    settings = config["data"]
    names = settings["classes"]
    if settings["name"] != "food101":
        raise ValueError("This experiment uses Food-101")
    if len(names) < 3 or len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
        raise ValueError("Choose at least three distinct Food-101 class names")

    counts = [settings[key] for key in (
        "train_per_class", "val_from_train_per_class", "val_from_test_per_class", "test_per_class")]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Split counts must be nonnegative integers")
    nt, nvt, nve, ne = counts
    if nt == 0 or ne == 0 or nvt + nve == 0:
        raise ValueError("Train, validation and test must be nonempty")

    source = FoodSource(settings["root"])
    unknown = set(names) - set(source.classes)
    if unknown:
        raise ValueError(f"Unknown Food-101 classes: {sorted(unknown)}")

    classes = [source.classes.index(name) for name in names]
    targets = np.asarray(source.targets)
    rng = np.random.default_rng(config["seed"])
    indices = {split: [] for split in ("train", "val", "test")}

    for label in classes:
        train_pool = rng.permutation(np.flatnonzero(targets[:source.train_size] == label))
        test_pool = rng.permutation(np.flatnonzero(targets[source.train_size:] == label) + source.train_size)
        if nt + nvt > len(train_pool) or ne + nve > len(test_pool):
            raise ValueError(f"Requested split exceeds available images for {source.classes[label]}")

        indices["train"].extend(train_pool[:nt].tolist())
        indices["val"].extend(train_pool[nt:nt + nvt].tolist())
        indices["val"].extend(test_pool[:nve].tolist())
        indices["test"].extend(test_pool[nve:nve + ne].tolist())

    for values in indices.values():
        rng.shuffle(values)

    size = config["image_size"]
    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    evaluation = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size), *normalize])
    training = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), *normalize])

    loaders = {}
    for offset, split in enumerate(indices):
        augmented = split == "train" and augment_train
        dataset = ImageSubset(source, indices[split], classes, training if augmented else evaluation, augmented)

        loaders[split] = DataLoader(
            dataset, batch_size=settings["batch_size"], shuffle=augmented,
            num_workers=settings["num_workers"], pin_memory=config["device"].startswith("cuda"),
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"] + offset))

    manifest = {
        "dataset": "Food101", "split_protocol": "train_holdout_and_test_holdout_v1",
        "seed": config["seed"], "classes": names, "indices": indices,
        "image_ids": {split: [source.image_ids[index] for index in values] for split, values in indices.items()},
        "official_train_size": source.train_size,
        "validation_per_class": {"official_train": nvt, "official_test": nve},
        "test_source": "official_test_after_validation_holdout", "image_size": size,
        "evaluation_transform": "resize256_bicubic_centercrop_imagenet_v1",
    }

    return DataBundle(loaders, names.copy(), manifest)


def dataset_table(bundle: DataBundle):
    import pandas as pd

    counts = {}
    for split, loader in bundle.loaders.items():
        dataset = loader.dataset
        labels = np.asarray(dataset.source.targets)[dataset.indices]
        counts[split] = [int((labels == label).sum()) for label in dataset.classes]

    return pd.DataFrame(counts, index=bundle.class_names).rename_axis("class")
