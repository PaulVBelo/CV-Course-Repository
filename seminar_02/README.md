# Семинар 2 — ViT, MAE и DINOv2 на Food-101

Сравнение linear probe и partial fine-tuning на 20 классах Food-101. Алгоритмы реализованы в Python-модулях; `experiment.ipynb` запускает эксперименты и показывает метрики, графики и примеры ошибок.

## Запуск

Готовый `experiment.ipynb` содержит исходные логи обучения и выполненные демонстрации. Для просмотра откройте его в VS Code: запуск ячеек не требуется.

Для нового эксперимента выберите kernel существующего `.venv/Scripts/python.exe`, установите `train_models = True` и выполняйте ячейки сверху вниз. После завершения сохраните notebook с outputs.

При `train_models = False` ячейка данных восстанавливает `probes` и `finetuned` из сохранённых результатов. Это позволяет отдельно выполнять нужные ячейки анализа. Повторное выполнение любой ячейки заменяет её outputs, поэтому исходные логи сохраняются, пока соответствующие ячейки обучения не выполняются заново.

В режиме просмотра `load_saved_experiment` проверяет manifest разбиения, параметры обучения и соответствие исходной головы FT выбранному LP. Версии библиотек и частота логирования не мешают чтению уже рассчитанных метрик. Графики и таблицы обновляются в `results/food101_20classes/presentation/`; t-SNE/UMAP используют кеш признаков, attention выполняет только inference исходной DINOv2.

При `train_models = True` первый полный запуск обучает три linear head по 20 эпох, затем три модели по 40 эпох с размороженными двумя последними блоками. Fine-tuning получает лучшую голову соответствующего linear probe автоматически. Модели работают последовательно, GPU освобождается между запусками.

В режиме обучения `run_experiment` повторно использует завершённый результат при точном совпадении конфигурации, окружения и исходной головы. `reuse=False` запускает новое обучение. Прерванный эксперимент начинается заново: восстановление состояния optimizer/scheduler не реализовано. Checkpoints сохраняют лучшую модель для inference и перехода от LP к FT.

CLI из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m seminar_02.train_linear_probe --backbone dinov2
.\.venv\Scripts\python.exe -m seminar_02.finetune --backbone dinov2 --probe-checkpoint checkpoints/seminar_02/food101_20classes/<run_id>/best.pt
```

Ключи backbone: `supervised`, `mae`, `dinov2`. Обе команды принимают `--config путь.yaml`; Python-файлы также поддерживают прямой запуск. Импорт модулей не скачивает данные, не загружает модели и не запускает обучение.

## Данные и разбиение

Food-101 содержит 101 категорию еды, по 750 официальных train и 250 test изображений каждого класса. Используем фиксированные 20 категорий; их названия и порядок записаны в `config.yaml`:

apple_pie, beef_carpaccio, bibimbap, caesar_salad, cheesecake, chicken_curry, chocolate_cake, club_sandwich, donuts, dumplings, french_fries, fried_rice, grilled_salmon, hamburger, ice_cream, omelette, pizza, ramen, steak, sushi.

| Наше разбиение | Из официального train на класс | Из официального test на класс | Всего на класс | Всего для 20 классов |
|---|---:|---:|---:|---:|
| Train | 700 | 0 | 700 | 14 000 |
| Validation | 50 | 50 | 100 | 2 000 |
| Test | 0 | 200 | 200 | 4 000 |

Имена параметров: `train_per_class`, `val_from_train_per_class`, `val_from_test_per_class`, `test_per_class`. Классы задаются строками в `data.classes`. Разбиение стратифицировано и фиксируется seed до обучения. Все три итоговых набора попарно не пересекаются. Перенесённые 50 test-изображений используются только для validation и исключены из итоговой оценки. Поэтому в отчёте указываем собственный протокол 700/100/200, а не стандартные benchmark-результаты Food-101.

`FoodSource` объединяет два официальных источника в единое пространство индексов. Это позволяет validation содержать фотографии обоих источников. Manifest сохраняет идентификаторы файлов, индексы, порядок классов, seed, происхождение validation и preprocessing. Одна и та же фотография не может попасть в разные итоговые наборы из-за совпадения локальных индексов источников.

Датасет скачивается автоматически через установленный `torchvision.datasets.Food101`. Загрузчик получает официальный архив всего Food-101, около 5 GB, и распаковывает его в `data/food-101`; отбор 20 классов происходит после загрузки. Для архива и распакованных фотографий нужно примерно 10 GB свободного места. Повторные запуски используют локальные файлы. Дополнительная библиотека `datasets` не нужна.

Исходные фотографии имеют разные пропорции и максимальную сторону до 512 px. Для LP, validation и test используем Resize(256, bicubic), CenterCrop(224), ToTensor и ImageNet mean/std. Если короткая сторона меньше 256, Resize увеличивает её; такие изображения остаются в фиксированной выборке. Для train в FT применяем RandomResizedCrop(224, scale=0.7..1.0) и horizontal flip. Validation/test детерминированы, случайные аугментации не кешируются.

Проверка всех 20 000 файлов текущей конфигурации: у 19 989 короткая сторона не меньше 256 px, у 11 — меньше 256, и только у одного — меньше 224. Медиана ширины и высоты — 512 px. Поэтому стандартный evaluation preprocessing почти всегда уменьшает исходные фотографии.

Официальный train содержит шумные метки, test проверен авторами вручную. Наша validation сочетает 50 фотографий из каждого источника. По умолчанию `num_workers=0` подходит для Windows/Jupyter; при запуске CLI число workers можно увеличить.

## Backbone и признаки

| Ключ | Checkpoint timm | Патч / токены при 224 | Признак |
|---|---|---|---|
| `supervised` | `vit_base_patch16_224.orig_in21k_ft_in1k` | 16 / 196 + CLS | Нормализованный CLS, 768 |
| `mae` | `vit_base_patch16_224.mae` | 16 / 196 + CLS | CLS encoder, 768 |
| `dinov2` | `vit_base_patch14_dinov2.lvd142m` | 14 / 256 + CLS | Нормализованный CLS, 768 |

Supervised ViT обучен с метками ImageNet-21k и затем ImageNet-1k. MAE обучался восстанавливать скрытые патчи; здесь используется encoder целого изображения без masking и decoder. DINOv2 извлекает признаки после self-supervised предобучения. SSL-предобучение в семинаре не повторяется.

Модели загружаются через `timm.create_model(..., pretrained=True, num_classes=0, img_size=224, global_pool="token")`. Timm адаптирует positional embeddings DINOv2 к сетке 224. Во всех моделях используем CLS одинаковой размерности. Mean-pooling патчей MAE можно исследовать отдельным экспериментом.

Supervised checkpoint ожидает mean/std=(0.5,0.5,0.5): адаптер внутри backbone преобразует ImageNet-нормализованный вход в эту шкалу. MAE и DINOv2 работают с ImageNet mean/std. Загрузка pretrained checkpoint обязательна: случайная инициализация при ошибке загрузки не подставляется.

## Linear probe

Для изображения x вычисляем фиксированный вектор z=f(x). Обучается линейная голова logits=Wz+b с cross-entropy. Backbone заморожен, находится в eval; AdamW обновляет только голову: lr=1e-3, weight_decay=1e-4, 20 эпох.

Encoder вычисляет признаки один раз с no_grad и AMP. Кеш хранит CPU FP32-тензоры `[N,768]`, точность исходного вычисления соответствует AMP BF16. Голова обучается в FP32 на готовых признаках; encoder освобождает GPU. Кеш всех трёх моделей на 20 000 фотографий занимает около 176 MiB без метаданных.

Ключ кеша включает SHA256 весов, manifest данных, preprocessing, версии torch/timm/torchvision, устройство и AMP dtype. Последовательный sampler и отсутствие случайных аугментаций проверяются перед кешированием. Для t-SNE/UMAP загружается только test-кеш. Изменение классов, разбиения или весов создаёт новый кеш и исключает использование несовместимой LP-головы.

## Partial fine-tuning

Обучаются два последних transformer blocks, финальная LayerNorm и linear head. Замороженный префикс остаётся в eval. Его параметры и вход не требуют градиентов, поэтому autograd начинает хранить граф с первого обучаемого блока.

Параметры по умолчанию:

- Верхний блок: lr=5e-5; предыдущий: 5e-5 × 0.65=3.25e-5.
- Финальная LayerNorm: lr=5e-5; голова: lr=5e-4.
- AdamW weight decay=0.01 для матриц; bias и одномерные параметры исключены из decay.
- 40 эпох: 4 эпохи warmup, далее cosine decay до нуля. Scheduler обновляется после optimizer step.
- Microbatch=8 и accumulation_steps=8: эффективный batch 64.
- AMP BF16 и gradient clipping по норме 1.0. Gradient checkpointing доступен в конфигурации и по умолчанию выключен.

Накопление градиентов суммирует cross-entropy по изображениям. Перед обновлением выполняются unscale, деление градиентов на фактическое число изображений в окне, clipping и optimizer step. Неполный последний microbatch получает корректный вес. FP16 доступен через `amp_dtype: float16` с GradScaler; при пропуске optimizer step scheduler также не продвигается.

BF16 уменьшает память активаций, сохраняя широкий диапазон значений. Параметры и AdamW states остаются FP32. При OOM уменьшите microbatch до 4 и увеличьте accumulation до 16. Для прежнего effective batch можно использовать 2 и 32. Новые CUDA-библиотеки, xFormers или переустановка PyTorch не требуются.

`probe_checkpoint=None` создаёт новую голову. В notebook передаётся лучший LP checkpoint той же модели и того же разбиения; совместимость проверяется по метаданным и хешу pretrained весов.

## Метрики и анализ

Loss усредняется по изображениям. Accuracy — доля верных ответов. Macro-F1 усредняет F1 по всем 20 классам с `zero_division=0`. Train-метрики собираются во время обновления модели; validation — после эпохи. FT train дополнительно содержит аугментации.

Checkpoint выбирается по максимальному validation macro-F1; при равенстве остаётся более ранняя эпоха. Итоговый test выполняется после восстановления лучшего checkpoint. Настройки обучения выбираются по validation.

Notebook показывает и сохраняет:

- Таблицы LP/FT и classification reports: precision, recall, F1 и support по каждому блюду.
- Train/validation кривые loss, accuracy и macro-F1 с отметкой выбранной эпохи.
- Разницу LP/FT в процентных пунктах и confusion matrices: два метода рядом для каждого backbone.
- FP/FN трёх классов с наибольшим числом пропусков; наиболее уверенные ошибки модели, выбранной по validation macro-F1, с softmax-вероятностью в заголовке.
- t-SNE и UMAP на одних и тех же 1 000 test-фотографий. Метрики классификации рассчитаны по всем 4 000 test-фотографиям.
- DINOv2 attention rollout для двух изображений.
- Числовой итог, максимальный прирост от fine-tuning и самую частую ошибку выбранной модели.

Для проекций признаки L2-нормируются и сжимаются PCA до 50 измерений. UMAP работает с фиксированным seed и n_jobs=1. Сравниваем смешение классов и компактность групп внутри панелей; связь с качеством классификации проверяем по confusion matrices.

Rollout получает настоящие attention probabilities через hooks на `attn_drop` в eval с временным `fused_attn=False`. Усредняем головы, добавляем identity, нормируем строки и последовательно перемножаем матрицы. CLS→patch строка преобразуется в сетку и накладывается на изображение. Hooks и режимы attention восстанавливаются после расчёта. Rollout показывает распространение внимания через слои, а не причинный вклад пикселей в решение классификатора.

## Файлы и API

| Файл | Основные функции и результат |
|---|---|
| `__init__.py` | Обозначает пакет без действий при импорте. |
| `runtime.py` | `load_config(path=None) -> dict`, `seed_everything(seed)`, `environment_info() -> dict`; AMP, пути запусков и сохранение JSON. Пути YAML разрешаются относительно корня курса. |
| `data.py` | `FoodSource(root)` объединяет официальные источники; `ImageSubset` применяет преобразования и переводит метки в 0..C-1. `create_dataloaders(config, augment_train=False) -> DataBundle` возвращает loaders, class_names и manifest; `dataset_table(bundle) -> DataFrame` считает размеры. |
| `models.py` | `load_backbone(name, device="cpu") -> Backbone`, forward `[B,3,224,224] -> [B,768]`; `ImageClassifier(backbone, num_classes)` добавляет голову; `configure_partial_finetuning(model, blocks)` размораживает верхние блоки. |
| `features.py` | `extract_features(backbone, loader, config) -> (features, labels)` возвращает CPU-тензоры. `cached_features(backbone, bundle, config, splits=None)` возвращает словарь TensorDataset. `comparison_features(config, bundle)` загружает test-признаки трёх моделей и общие метки. |
| `engine.py` | `fit(...) -> dict` выполняет обучение, validation, выбор checkpoint и test; `layerwise_parameter_groups` создаёт AdamW-группы; `warmup_cosine` возвращает scheduler. |
| `train_linear_probe.py` | `train_linear_probe(config, backbone_name="dinov2", smoke=False) -> dict`; `load_probe_head` проверяет совместимость сохранённой головы. Есть CLI. |
| `finetune.py` | `finetune_model(config, backbone_name="dinov2", probe_checkpoint=None, smoke=False) -> dict`; `load_classifier(checkpoint, device="cpu")` восстанавливает LP или FT классификатор для изображений. Есть CLI. |
| `experiments.py` | `run_experiment(config, backbone_name, mode, probe_checkpoint=None, reuse=True)` запускает или читает эксперимент; `load_saved_experiment(config, backbone_name, mode, manifest, probe_checkpoint=None)` только читает готовый результат, без запуска обучения; `results_table`, `compare_results`, `summarize_results` строят сводки. |
| `evaluate.py` | `evaluate_model(model, loader, device, config, num_classes) -> dict`: метрики, targets, predictions, probabilities и confusion matrix. `classification_reports(results) -> DataFrame`; `error_analysis(result, class_names) -> DataFrame`; `comparison_table` собирает завершённые запуски. |
| `visualize.py` | `plot_samples`, `plot_history`, `plot_learning_curves`, `plot_confusion_matrix`, `plot_confusion_matrices`, `plot_metric_comparison`, `plot_errors`, `plot_attention_examples` возвращают Figure и принимают `path` для сохранения. `compare_embeddings(...) -> (Figure, coordinates)`; `attention_rollout(backbone, image) -> Tensor[H,W]`. |
| `sanity.py` | `backbone_checks(config, bundle)` возвращает компактную таблицу forward и overfit одного батча; `overfit_head` обучает маленькую голову. Дополнительный `run_smoke_tests` нужен только для проверки изменений пайплайна. |
| `test_protocol.py` | Unittest-проверки разбиения, источников validation, кеша, заморозки, accumulation и scheduler. |
| `config.yaml` | Классы, размеры наборов, seed, оптимизация, память и каталоги. |
| `experiment.ipynb` | Последовательный интерфейс для запуска и анализа. Алгоритмы в нём не дублируются. |

## Результаты и публикация

- `seminar_02/results/food101_20classes/<run_id>/`: config.json, environment.json, metrics.json, history.csv, predictions.csv, errors.csv, classification reports и графики.
- `seminar_02/results/food101_20classes/presentation/`: сводные CSV, PNG и summary.md из notebook.
- `checkpoints/seminar_02/food101_20classes/<run_id>/best.pt`: лучшие модели, исключены из Git.
- `.cache/seminar_02/food101_20classes/`: frozen features, исключены из Git.
- `data/food-101/`: исходные фотографии и метаданные, исключены из Git.

Перед публикацией выполните notebook на окончательном YAML и сохраните настоящие outputs. Проверьте шесть завершённых запусков и наличие сводных таблиц. В Git можно добавить код, YAML, notebook, README и небольшие CSV/TXT/JSON/PNG. `.venv`, датасеты, веса, checkpoints и кеши должны оставаться исключёнными. Push выполняется отдельно.

В рабочем notebook нет результатов Food-101 до полного обучения. Проверочные запуски не используются в сравнительной таблице.

## Окружение и воспроизводимость

Используется существующее `.venv`: Python 3.14.8, torch 2.14.0+cu130, torchvision 0.29.0+cu130, timm 1.0.30, RTX 5060 Ti 8 GB. Python, CUDA, PyTorch и общий requirements.txt не переустанавливаются.

Фиксируются random, NumPy, PyTorch, CUDA, DataLoader generators и worker seeds. `cudnn.benchmark` выключен. Режим deterministic algorithms включён с warn_only=True; memory-efficient attention backward в текущей CUDA-сборке предупреждает о недетерминированном kernel. Поэтому сохраняем версии окружения, seed, manifest и хеш весов каждого запуска, а побитовую идентичность GPU-вычислений не предполагаем.

Проверки протокола:

```powershell
.\.venv\Scripts\python.exe -m unittest seminar_02.test_protocol -v
```

Семь тестов проверяют разбиения на 3 и 20 классах, происхождение validation из обоих источников, отсутствие пересечений, ошибки конфигурации, кеш, lazy imports, gradient accumulation, заморозку и scheduler.

На реальном Food-101 проверены все 20 000 выбранных файлов и разбиение 700/100/200. Notebook выполнен целиком на временном подмножестве: по 40 train, validation и test фотографий, по одной эпохе LP и FT для каждой из трёх pretrained моделей. Прошли forward, overfit головы, обучение с AMP и accumulation, сохранение checkpoints и classification reports, t-SNE, UMAP и attention rollout. Проверочные файлы удалены; рабочий notebook остаётся без outputs до полного запуска на 14 000 / 2 000 / 4 000 изображениях.

## Источники

- Материалы семинара: Seminar_2.md и Lecture_2.pdf.
- [Food-101: описание, официальный архив и статья](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Torchvision Food101](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html)
- [Supervised ViT checkpoint](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k)
- [MAE checkpoint](https://huggingface.co/timm/vit_base_patch16_224.mae)
- [DINOv2 checkpoint](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m)
- [PyTorch AMP и accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
