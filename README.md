# CV-Course

Общий репозиторий курса Computer Vision.

## Семинар 2

[Экспериментальный notebook](seminar_02/experiment.ipynb) · [Инструкция и API](seminar_02/README.md) · [Задание](seminar_02/Seminar_2.md)

ViT, MAE и DINOv2: linear probe, partial fine-tuning и анализ результатов на 20 классах Food-101. На класс используются 700 train, 100 validation (50 из официального train + 50 из test) и 200 test изображений.

Откройте notebook в VS Code, выберите существующее `.venv`, перезапустите kernel и выполняйте ячейки сверху вниз. Первый полный запуск обучает три linear probe по 20 эпох и три partial fine-tuning по 40 эпох. Повторное выполнение использует завершённые запуски с той же конфигурацией.

Classification reports, кривые обучения, confusion matrices, проекции и attention выводятся в notebook и сохраняются в `seminar_02/results/food101_20classes/presentation/`. AMP BF16 и накопление градиентов поддерживают effective batch 64 при microbatch 8.

Алгоритмы находятся в Python-модулях, параметры — в `seminar_02/config.yaml`, зависимости — в общем `requirements.txt`.
