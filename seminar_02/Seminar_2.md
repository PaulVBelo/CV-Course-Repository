
---

# Семинар 2 — ViT как бэкбон + MAE/DINOv3 как «универсальные фичи»

Цель семинара: научиться использовать предобученные модели (supervised ViT, MAE, DINOv2) как фиче-экстракторы, сравнить *linear probe* и *fine-tuning*, провести sanity-checks и оформить минимальный репозиторий для reproducibility.

---

## S0. Подготовка и sanity-checks

**Шаги:**

1. Зафиксировать сиды для reproducibility (`torch.manual_seed`, `numpy.random.seed`).
2. Проверить загрузку датасета (мини-датасет с ≥3 классами).
3. Сформировать `DataLoader` с нормализацией под ImageNet.
4. Прогнать один батч через предобученный ViT (supervised, MAE, DINOv2).
5. Сделать «overfit на 1–2 батчах» для линейной головы, чтобы убедиться в корректности пайплайна.

**Контрольный чек-лист:**

* модель загружается и работает на GPU,
* batch проходит через энкодер,
* размерность признаков совпадает (например, ViT-B/16 → 768),
* линейная голова сходится на 1–2 батчах.

---

## S1. Linear probe на предобученных фичах

**Шаги:**

1. Заморозить энкодер (`requires_grad_(False)`).
2. Обучить линейную голову поверх DINOv2 и MAE.
3. Для контроля — обучить ту же голову на supervised ViT-B/16 (замороженный).
4. Сравнить точность (accuracy/F1) и построить confusion matrix.

**Минимальный результат:** голова сходится, метрики > случайных.
**Целевой результат:** DINOv2/MAE дают более высокие метрики, чем supervised ViT.

**Кодовая заготовка:**

```python
import torch, torch.nn as nn, torch.nn.functional as F
import timm

# Загружаем предобученный DINOv2-бэкбон
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)

# Линейная голова
head = nn.Linear(backbone.num_features, num_classes).cuda()
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(20):
    head.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            feats = backbone(x)
        logits = head(feats)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
```

---

## S2. Fine-tuning с частичным размораживанием

**Шаги:**

1. Разморозить верхние 1–2 блока энкодера.
2. Настроить *layer-wise LR decay* (например, α=0.65).
3. Обучить модель 30–50 эпох с warmup+cosine scheduler.
4. Отслеживать переобучение (train–val gap, sanity-split).

**Минимальный результат:** метрики не хуже, чем у linear probe.
**Целевой результат:** прирост accuracy/F1 на 5–10% при правильном тюнинге.

---

## S3. Аналитика и визуализация

**Шаги:**

1. Построить confusion matrix.
2. Визуализировать эмбеддинги через t-SNE/UMAP (supervised vs DINOv2 vs MAE).
3. Сделать attention rollout для пары изображений (у DINOv2 токены «схватывают» объекты).
4. Обсудить false positives и false negatives, обратить внимание на хвостовые классы.

**Минимальный результат:** хотя бы одна t-SNE/UMAP-проекция и одна attention-карта.
**Целевой результат:** качественное обсуждение того, чем отличаются supervised и SSL-признаки.

---

## S4. Репозиторий и reproducibility

**Шаги:**

1. Выложить на GitHub минимальный рабочий код.
2. Обязательно добавить:

   * `requirements.txt` с версиями,
   * `train_linear_probe.py` и `finetune.py`,
   * конфиги (lr, wd, batch, α для LR decay),
   * логи TensorBoard/CSV,
   * README с инструкцией по запуску.
3. Сравнительная таблица метрик (supervised ViT, MAE, DINOv2).

**Минимальный результат:** код запускается и воспроизводит обучение linear probe.
**Целевой результат:** полный reproducible репозиторий с линейной и fine-tuning кривыми, визуализациями и метриками.

---

### Что положить в репозиторий (итоговый чек-лист)

* Код: `train_linear_probe.py`, `finetune.py`.
* Конфиги: YAML/JSON с гиперпараметрами.
* Логи: TensorBoard/CSV.
* Визуализации: confusion matrix, t-SNE/UMAP.
* Таблица метрик: supervised ViT vs MAE vs DINOv2.
* README с инструкциями по воспроизведению.

---
