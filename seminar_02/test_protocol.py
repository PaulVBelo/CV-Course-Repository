import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from seminar_02.data import create_dataloaders
from seminar_02.engine import fit, layerwise_parameter_groups, warmup_cosine
from seminar_02.features import cached_features
from seminar_02.models import ImageClassifier, configure_partial_finetuning
from seminar_02.runtime import load_config, seed_everything


class FakeFood:
    classes = sorted(load_config()['data']['classes'])

    def __init__(self, root, split, download):
        self._images_folder = Path(root) / 'food-101/images'
        self._labels = np.repeat(np.arange(len(self.classes)), 30).tolist()
        self._image_files = [self._images_folder / self.classes[label] / f'{split}_{index}.jpg'
                             for index, label in enumerate(self._labels)]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return Image.fromarray(np.full((256, 320, 3), index % 255, dtype=np.uint8)), self._labels[index]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4

        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.encoder.norm = nn.LayerNorm(4)

    def forward(self, inputs):
        for block in self.encoder.blocks:
            inputs = block(inputs)
        return self.encoder.norm(inputs)


class ProtocolTests(unittest.TestCase):
    def test_imports_are_lazy(self):
        with patch('timm.create_model', side_effect=AssertionError('Model loaded during import')):
            for name in ('data', 'models', 'features', 'train_linear_probe', 'finetune', 'evaluate', 'visualize', 'sanity'):
                importlib.reload(importlib.import_module(f'seminar_02.{name}'))

    def test_split_disjoint_reproducible_and_balanced(self):
        names = load_config()['data']['classes']
        for classes in (names[:3], names):
            with self.subTest(classes=classes):
                config = load_config()
                config['data'].update(classes=classes, train_per_class=6, val_from_train_per_class=3,
                                      val_from_test_per_class=2, test_per_class=4)

                with patch('seminar_02.data.datasets.Food101', FakeFood):
                    first = create_dataloaders(config)
                    second = create_dataloaders(config, augment_train=True)

                self.assertEqual(first.manifest, second.manifest)
                split_ids = first.manifest['image_ids']
                for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                    self.assertFalse(set(split_ids[left]) & set(split_ids[right]))

                for split, count in (('train', 6), ('val', 5), ('test', 4)):
                    dataset = first.loaders[split].dataset
                    self.assertEqual(len(dataset), count * len(classes))
                    labels = [dataset[i][1] for i in range(len(dataset))]
                    self.assertEqual(np.bincount(labels).tolist(), [count] * len(classes))

                validation = first.loaders['val'].dataset
                targets = np.asarray(validation.source.targets)[validation.indices]
                from_train = np.asarray(validation.indices) < validation.source.train_size
                for label in validation.classes:
                    self.assertEqual(int(((targets == label) & from_train).sum()), 3)
                    self.assertEqual(int(((targets == label) & ~from_train).sum()), 2)

    def test_invalid_food_splits_are_rejected(self):
        config = load_config()
        config['data'].update(train_per_class=28, val_from_train_per_class=3,
                              val_from_test_per_class=2, test_per_class=4)
        with patch('seminar_02.data.datasets.Food101', FakeFood):
            with self.assertRaisesRegex(ValueError, 'exceeds available'):
                create_dataloaders(config)

            config['data'].update(train_per_class=6, val_from_test_per_class=28)
            with self.assertRaisesRegex(ValueError, 'exceeds available'):
                create_dataloaders(config)

            config['data'].update(val_from_test_per_class=2)
            config['data']['classes'][0] = 'unknown_food'
            with self.assertRaisesRegex(ValueError, 'Unknown Food-101'):
                create_dataloaders(config)

    def test_cache_reuses_requested_split_and_invalidates_changed_protocol(self):
        config = load_config()
        config.update(device='cpu', amp=False)
        backbone = TinyBackbone()
        backbone.requires_grad_(False)
        backbone.name = 'test'
        backbone.fingerprint = lambda: 'fixed_test_weights'

        dataset = TensorDataset(torch.randn(6, 4), torch.arange(6) % 3)
        dataset.augmented = False
        bundle = SimpleNamespace(loaders={'test': DataLoader(dataset, batch_size=2)}, manifest={'seed': 42})
        extracted = (torch.randn(6, 4), torch.arange(6) % 3)

        with tempfile.TemporaryDirectory() as directory:
            config['cache_dir'] = directory
            with patch('seminar_02.features.extract_features', return_value=extracted) as extract:
                first = cached_features(backbone, bundle, config, splits=('test',))
                second = cached_features(backbone, bundle, config, splits=('test',))
                self.assertEqual(extract.call_count, 1)
                self.assertEqual(set(second), {'test'})
                torch.testing.assert_close(first['test'].tensors[0], second['test'].tensors[0])

                bundle.manifest = {'seed': 43}
                cached_features(backbone, bundle, config, splits=('test',))
                self.assertEqual(extract.call_count, 2)

    def test_partial_freezing_and_layer_decay(self):
        model = ImageClassifier(TinyBackbone(), 3)
        configure_partial_finetuning(model, 2)
        model.train()

        self.assertFalse(model.backbone.encoder.blocks[0].training)
        self.assertTrue(model.backbone.encoder.blocks[-1].training)

        F = torch.nn.functional
        F.cross_entropy(model(torch.randn(5, 4)), torch.tensor([0, 1, 2, 0, 1])).backward()

        self.assertTrue(all(p.grad is None for p in model.backbone.encoder.blocks[0].parameters()))
        self.assertTrue(all(p.grad is not None for p in model.backbone.encoder.blocks[-1].parameters()))

        settings = load_config()['finetune']
        groups = layerwise_parameter_groups(model, settings)
        rates = {name: group['lr'] for group in groups for name in group['names']}
        self.assertAlmostEqual(rates['backbone.encoder.blocks.1.weight'] / rates['backbone.encoder.blocks.2.weight'], 0.65)

    def test_uneven_accumulation_matches_full_batch(self):
        seed_everything(42)
        config = load_config()
        config.update(device='cpu', amp=False)

        inputs, labels = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
        dataset = TensorDataset(inputs, labels)
        initial = nn.Linear(4, 3)
        models = []

        with tempfile.TemporaryDirectory() as directory:
            for batch, accumulation in ((2, 3), (5, 1)):
                model = copy.deepcopy(initial)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                loader = DataLoader(dataset, batch_size=batch)

                output = Path(directory) / str(batch)
                output.mkdir()
                metadata = {'mode': 'linear_probe', 'class_names': ['a', 'b', 'c']}

                fit(model, dict(train=loader, val=loader, test=loader), optimizer, config,
                    {'epochs': 1, 'accumulation_steps': accumulation}, 3,
                    output / 'best.pt', output, metadata)
                models.append(model)

            for left, right in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    def test_scheduler_warmup_then_decay(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = warmup_cosine(optimizer, 6, 2)

        rates = []
        for _ in range(6):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        self.assertEqual(rates[:2], [0.5, 1.0])
        self.assertEqual(rates[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))


if __name__ == '__main__':
    unittest.main()
