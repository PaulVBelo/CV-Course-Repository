# Семинар 3. CLIP и SigLIP: классификация через текст и межмодальный поиск

**Курс:** компьютерное зрение, магистратура. **Продолжительность:** 90 минут. **Связь с лекцией 3:** совместное пространство изображения и текста, контрастивная и попарная логистическая цели, классификация без обучения на целевой разметке, поиск и адаптация.

Цель семинара — построить работающий эксперимент с двумя предобученными двухэнкодерными моделями и объяснить полученные результаты через устройство их представлений и функций потерь. В конце занятия студент должен уметь извлекать совместимые признаки изображения и текста, строить текстовый классификатор, измерять качество поиска и отличать относительную уверенность softmax от попарной оценки sigmoid.

Основной результат — один последовательно запускаемый ноутбук и небольшой отчёт. Обучение двух энкодеров с нуля не требуется. Все блоки Python ниже выполняются по порядку в одном ядре. Дополнительное задание с линейной головой выполняется после основной части.

## План занятия

| Этап | Время | Результат |
|---|---:|---|
| S0. Окружение и данные | 10 минут | Фиксированная выборка, процессоры, первая проверка |
| S1. Классификация CLIP и SigLIP | 25 минут | Признаки и сопоставимая таблица качества |
| S2. Формулировки и уверенность | 15 минут | Сравнение шаблонов и проверка зависимости от кандидатов |
| S3. Поиск по текстовому запросу | 20 минут | Галерея результатов, метрики поиска |
| S4. Функции потерь и выводы | 20 минут | Численные проверки CLIPLoss/SigLIPLoss и отчёт |

Загрузку весов и датасета желательно выполнить до занятия: время скачивания в эти 90 минут не включено. Преподаватель заранее проверяет доступ к Hugging Face и CIFAR-10. Если интернет недоступен, используются заранее загруженные кэши моделей и каталог `data`.

---

## S0. Подготовка и проверки

### Окружение

Основная конфигурация — Python 3.11, одна GPU уровня Colab T4, вычисления в FP32. На CPU задание также можно выполнить, уменьшив выборку. Две модели загружаются **последовательно**, признаки сохраняются на CPU. На Apple Silicon предусмотрен MPS; при неподдерживаемой операции выберите CPU.

Ниже зафиксирована версия Transformers, для которой написан интерфейс извлечения признаков. Это учебное окружение, а не требование использовать эти версии во всех проектах. В новом виртуальном окружении выполните:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 transformers==4.46.3 sentencepiece==0.2.0 protobuf==5.28.3 numpy==1.26.4 scikit-learn==1.5.2 matplotlib==3.9.2 pandas==2.2.3 pillow==11.0.0
```

В Colab используйте `%pip` вместо `python -m pip`; после установки перезапустите ядро. Не меняйте версию только `torch`, оставляя несовместимую `torchvision`. Код извлечения признаков ориентирован именно на Transformers 4.46.3: при переходе на другую версию необходимо проверить тип возвращаемого значения.

```python
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import transformers
from torchvision.datasets import CIFAR10
from transformers import AutoProcessor, CLIPModel, SiglipModel
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

OUT = Path('results_seminar3')
OUT.mkdir(exist_ok=True)
print('device:', device, 'torch:', torch.__version__)
print('transformers:', transformers.__version__)
```

### Данные и разделение

Берём CIFAR-10. Малое разрешение исходных изображений — ограничение эксперимента: увеличение до входного размера модели не восстанавливает утраченные детали. Используем 20 изображений каждого класса для проверки формулировок и 50 каждого класса для итогового теста. Выборка для проверки берётся из обучающей части CIFAR-10, тестовая — из официальной тестовой части.

На CPU можно установить `N_VAL = 5`, `N_TEST = 10`. Числа фиксируются до эксперимента; обе модели получают одни и те же индексы. Проверочная выборка используется для выбора шаблона, тестовая — один раз для итоговой оценки. Наличие изображений CIFAR-10 в неизвестных обучающих корпусах моделей здесь не проверяется; это учебная оценка переноса готовых моделей, а не доказательство отсутствия пересечения данных.

```python
N_VAL, N_TEST = (20, 50) if device.type != 'cpu' else (5, 10)
train_ds = CIFAR10(root='data', train=True, download=True)
test_ds = CIFAR10(root='data', train=False, download=True)
classes = train_ds.classes
# Имена классов оставляем английскими для базового эксперимента.

def balanced_indices(dataset, per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets)
    ids = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
        for c in range(len(classes))
    ])
    rng.shuffle(ids)
    return ids

val_ids = balanced_indices(train_ds, N_VAL, SEED)
test_ids = balanced_indices(test_ds, N_TEST, SEED + 1)
val_images = [train_ds[int(i)][0].convert('RGB') for i in val_ids]
test_images = [test_ds[int(i)][0].convert('RGB') for i in test_ids]
y_val = torch.tensor([train_ds.targets[int(i)] for i in val_ids])
y_test = torch.tensor([test_ds.targets[int(i)] for i in test_ids])

(OUT / 'split.json').write_text(json.dumps({
    'seed': SEED, 'val_source': 'CIFAR10/train',
    'test_source': 'CIFAR10/test',
    'val_ids': val_ids.tolist(), 'test_ids': test_ids.tolist(),
    'classes': classes,
}, indent=2), encoding='utf-8')
print(classes, len(val_images), len(test_images))
```

**Проверка:** показать несколько изображений, проверить соответствие меткам и убедиться, что представлены все десять классов. Нельзя подавать на вход обеим моделям произвольную общую нормализацию: у каждой модели используется собственный процессор.

---

## S1. Классификация CLIP и SigLIP без обучения

### Постановка

Для каждого класса создаём текстовое описание и получаем его нормированный вектор. Изображение относится к классу с максимальным скалярным произведением:

$$
\hat c(x)=\arg\max_c v(x)^\top u(t_c),\qquad \|v\|_2=\|u\|_2=1.
$$

Положительный общий масштаб логитов и общее смещение не меняют этот `argmax`. Однако они меняют значения sigmoid/softmax, поэтому для анализа уверенности мы их сохраняем.

Сравниваем `openai/clip-vit-base-patch32` и `google/siglip-base-patch16-224`. Это сравнение **готовых контрольных точек**, а не изолированное сравнение двух функций потерь: модели отличаются размером патча, предобучением и другими деталями. Нельзя приписывать всю разницу accuracy исключительно sigmoid loss.

### Извлечение и кэширование признаков

```python
MODEL_SPECS = {
    'CLIP': ('openai/clip-vit-base-patch32', CLIPModel),
    'SigLIP': ('google/siglip-base-patch16-224', SiglipModel),
}
TEMPLATES = {
    'name': '{}',
    'photo': 'a photo of a {}.',
    'sentence': 'This is a photo of a {}.',
}

@torch.no_grad()
def encode_images(model, processor, images, batch_size=16):
    chunks = []
    for start in range(0, len(images), batch_size):
        inputs = processor(images=images[start:start + batch_size],
                           return_tensors='pt').to(device)
        features = model.get_image_features(**inputs)
        chunks.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks)

@torch.no_grad()
def encode_texts(model, processor, texts, kind):
    # SigLIP обучался с дополнением текста до фиксированной длины.
    padding = 'max_length' if kind == 'SigLIP' else True
    inputs = processor(text=texts, padding=padding, truncation=True,
                       return_tensors='pt').to(device)
    features = model.get_text_features(**inputs)
    return F.normalize(features.float(), dim=-1).cpu()

cache = {}
for kind, (model_id, model_class) in MODEL_SPECS.items():
    print('Loading', model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id).to(device).eval()
    model.requires_grad_(False)
    image_val = encode_images(model, processor, val_images)
    image_test = encode_images(model, processor, test_images)
    prototypes = {
        name: encode_texts(model, processor,
                           [template.format(c) for c in classes], kind)
        for name, template in TEMPLATES.items()
    }
    # Сначала нормируем каждый текст, затем усредняем и снова нормируем.
    prototypes['ensemble'] = F.normalize(
        torch.stack(list(prototypes.values())).mean(dim=0), dim=-1)
    scale = model.logit_scale.detach().float().exp().cpu()
    bias = (model.logit_bias.detach().float().cpu()
            if kind == 'SigLIP' else torch.tensor(0.0))

    # Сверяем ручные логиты с forward модели на тех же входах.
    texts = [TEMPLATES['photo'].format(c) for c in classes]
    inputs = processor(images=test_images[:2], text=texts,
                       padding='max_length' if kind == 'SigLIP' else True,
                       truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        official = model(**inputs).logits_per_image.float().cpu()
    manual = scale * image_test[:2] @ prototypes['photo'].T + bias
    assert torch.allclose(official, manual, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(image_test).all()
    assert torch.allclose(image_test.norm(dim=-1),
                          torch.ones(len(test_images)), atol=1e-4)
    assert image_test.shape[1] == prototypes['photo'].shape[1]

    cache[kind] = {
        'val': image_val, 'test': image_test, 'texts': prototypes,
        'scale': scale, 'bias': bias,
        'model_id': model_id,
        'revision': getattr(model.config, '_commit_hash', None),
    }
    torch.save(cache[kind], OUT / f'{kind}_features.pt')
    print(kind, image_test.shape, 'scale=', scale.item(),
          'bias=', bias.item())
    del model, processor, inputs, official
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
```

Если проверка логитов не проходит, сначала проверяем версии, `eval()`, padding и нормализацию. На разных устройствах возможны небольшие численные расхождения, но крупное расхождение нельзя скрывать повышением допуска. Если возникает нехватка памяти, уменьшаем `batch_size` до 4; смешанную точность в базовом опыте не включаем.

### Базовая таблица на проверочной выборке

```python
def classification_metrics(features, prototypes, labels):
    pred = (features @ prototypes.T).argmax(dim=-1).numpy()
    truth = labels.numpy()
    return {
        'accuracy': accuracy_score(truth, pred),
        'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
    }

val_rows = []
for kind, item in cache.items():
    for prompt, texts in item['texts'].items():
        val_rows.append({'model': kind, 'prompt': prompt,
                         **classification_metrics(item['val'], texts, y_val)})
val_table = pd.DataFrame(val_rows)
val_table.to_csv(OUT / 'validation.csv', index=False)
print(val_table.to_string(index=False))
```

**Задание:** объяснить формы тензоров `image_test`, `prototypes['photo']`, `image_test @ prototypes['photo'].T`. Проверить, что качество осмысленно относительно случайного выбора из десяти классов. Значение выше 10% само по себе не доказывает отсутствие ошибки в эксперименте; обязательно осмотреть предсказания.

**Минимальный результат:** обе модели возвращают конечные нормированные признаки; ручные логиты совпадают с выходом модели; таблица качества построена на одних и тех же примерах.

---

## S2. Формулировки, ансамбль описаний и уверенность

### Выбор текстовых прототипов

Сравниваем короткое название, два предложения и усреднение трёх описаний. Лучший вариант выбирается **по проверочной выборке**. При равенстве accuracy сохраняется первый вариант в порядке таблицы; правило не меняется после просмотра теста.

```python
selected = {}
test_rows = []
for kind, item in cache.items():
    candidates = val_table[val_table.model == kind]
    best_row = candidates.loc[candidates.accuracy.idxmax()]
    chosen = best_row['prompt']
    selected[kind] = chosen
    for prompt in dict.fromkeys(['photo', chosen]):
        test_rows.append({'model': kind, 'prompt': prompt,
                          'selected_on_val': prompt == chosen,
                          **classification_metrics(
                              item['test'], item['texts'][prompt], y_test)})

test_table = pd.DataFrame(test_rows)
test_table.to_csv(OUT / 'test.csv', index=False)
print(test_table.to_string(index=False))

for kind, item in cache.items():
    texts = item['texts'][selected[kind]]
    prediction = (item['test'] @ texts.T).argmax(dim=-1).numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test.numpy(), prediction, display_labels=classes,
        normalize='true', xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(f'{kind}: {selected[kind]}')
    fig.tight_layout()
    fig.savefig(OUT / f'{kind}_confusion.png', dpi=150)
    plt.show()
```

**Разбор:** изменились ли ошибки между `cat` и `dog`, `automobile` и `truck`? Одинаково ли действуют шаблоны на обе модели? Почему ансамбль может не улучшить результат? Не требуется получить заранее заданный прирост: отрицательный результат при корректном протоколе тоже является результатом.

### Что происходит при изменении списка кандидатов

Для одного изображения сравните первые три текстовых кандидата со всеми десятью. Уже вычисленные логиты первых трёх пар остаются одинаковыми; изменяется только нормировка softmax.

```python
for kind, item in cache.items():
    proto = item['texts']['photo']
    logits = item['scale'] * item['test'][0:1] @ proto.T + item['bias']
    small_softmax = logits[:, :3].softmax(dim=-1)
    full_softmax = logits.softmax(dim=-1)[:, :3]
    small_sigmoid = logits[:, :3].sigmoid()
    full_sigmoid = logits.sigmoid()[:, :3]
    assert torch.allclose(small_sigmoid, full_sigmoid)
    print(kind, 'true:', classes[y_test[0].item()])
    print('classes:', classes[:3])
    print('softmax, 3 candidates:', small_softmax.tolist())
    print('softmax, 10 candidates:', full_softmax.tolist())
    print('sigmoid, same pairs:', small_sigmoid.tolist())
```

Здесь обе функции намеренно применяются к обеим матрицам, чтобы изучить их математическое поведение. **Sigmoid логитов CLIP не является штатной оценкой соответствия этой модели.** Для CLIP softmax описывает выбор внутри предоставленного набора; для SigLIP используется sigmoid логитов с обученными масштабом и смещением. Ни один из этих результатов без дополнительной проверки не следует трактовать как откалиброванную вероятность реального события на CIFAR-10.

**Вопросы:** почему порядок классов не меняется при положительном общем масштабе? Почему добавление нерелевантного кандидата может уменьшить softmax-оценку? Почему маленькое значение sigmoid не означает, что нужно обязательно отвергнуть изображение по универсальному порогу 0.5?

---

## S3. Межмодальный поиск

### Индекс и запрос

Используем тестовые изображения как небольшой поисковый индекс. Текстовые векторы уже закэшированы; повторного прохода через визуальный энкодер не нужно. Для каждого из десяти классов выполняем текстовый запрос и ранжируем изображения по косинусному сходству.

В этом упражнении релевантность определяется **классом изображения**, а не уникальной парой «изображение — подпись». Поэтому основная метрика — Precision@5. Дополнительно считаем долю релевантных изображений, найденных в первых пяти результатах, и долю запросов хотя бы с одним попаданием. Последняя называется Hit@5 и не подменяет Recall@5.

$$
P@k(q)=\frac{|Top_k(q)\cap Rel(q)|}{k},\qquad
R@k(q)=\frac{|Top_k(q)\cap Rel(q)|}{|Rel(q)|}.
$$

При 50 релевантных изображениях класса максимально достижимый Recall@5 равен 0.1. Это свойство постановки, а не признак плохой модели. Мы усредняем метрики по десяти запросам.

```python
def retrieval_metrics(image_features, text_features, labels, k=5):
    scores = text_features @ image_features.T  # [10, N]
    k = min(k, image_features.shape[0])
    top = scores.topk(k, dim=-1).indices
    relevance = labels[top].eq(torch.arange(len(classes))[:, None])
    found = relevance.sum(dim=1).float()
    totals = torch.bincount(labels, minlength=len(classes)).float()
    return {
        f'precision@{k}': (found / k).mean().item(),
        f'recall@{k}': (found / totals).mean().item(),
        f'hit@{k}': relevance.any(dim=1).float().mean().item(),
    }, top

retrieval_rows = []
for kind, item in cache.items():
    texts = item['texts'][selected[kind]]
    metrics, top = retrieval_metrics(item['test'], texts, y_test)
    retrieval_rows.append({'model': kind, **metrics})
    query_class = classes.index('dog')
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    for ax, idx in zip(axes, top[query_class].tolist()):
        ax.imshow(test_images[idx])
        ax.set_title(classes[y_test[idx].item()])
        ax.axis('off')
    fig.suptitle(f'{kind}: query = dog')
    fig.tight_layout()
    fig.savefig(OUT / f'{kind}_search_dog.png', dpi=150)
    plt.show()

retrieval_table = pd.DataFrame(retrieval_rows)
retrieval_table.to_csv(OUT / 'retrieval.csv', index=False)
print(retrieval_table.to_string(index=False))
```

**Задание:** повторить визуализацию для двух других классов. Найти минимум три ошибочных результата и объяснить, какие визуальные признаки могли повлиять на ранжирование. Отдельно отметить случаи, когда низкое разрешение не позволяет уверенно судить о содержимом.

**Проверка понимания:** почему добавление общего смещения SigLIP не меняет порядок результатов? Почему нельзя искать по текстовым векторам SigLIP в индексе, построенном CLIP, даже если размерности случайно совпадут?

**Дополнительный опыт:** взять 10 собственных фотографий, вручную написать к каждой две корректные подписи и добавить трудные ложные подписи с заменой цвета, количества или отношения объектов. Для этого заново загрузить одну модель, закодировать новые тексты функцией `encode_texts` и изображения функцией `encode_images`. Составить отдельную таблицу истинных соответствий; не считать автоматически все внедиагональные пары отрицательными. Оценить ранжирование, а ошибки композиции описать отдельно от узнавания класса. Этот опыт не смешивается с таблицей CIFAR-10.

---

## S4. Функции потерь: что именно меняется в обучении

### Две устойчивые реализации

Ниже вход `logits` — матрица размера $B\times B$, где диагональ соответствует положительным парам. CLIP использует симметричную перекрёстную энтропию. SigLIP использует бинарную логистическую цель с нормировкой на $B$, как в лекции. Это **не** `mean()` по всем $B^2$ элементам.

$$
\mathcal L_{CLIP}=\tfrac12\left[CE(S,\operatorname{arange}(B))+
CE(S^\top,\operatorname{arange}(B))\right],
$$

$$
\mathcal L_{SigLIP}=\frac1B\sum_{i,j}
\operatorname{softplus}(-z_{ij}S_{ij}),\qquad
z_{ij}=\begin{cases}+1,&i=j,\\-1,&i\ne j.\end{cases}
$$

```python
def clip_loss(logits):
    b = logits.shape[0]
    assert logits.ndim == 2 and logits.shape[1] == b
    target = torch.arange(b, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.T, target))

def siglip_loss(logits):
    b = logits.shape[0]
    assert logits.ndim == 2 and logits.shape[1] == b
    target = torch.eye(b, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits, target, reduction='sum') / b

B = 4
zero = torch.zeros(B, B)
good = torch.full((B, B), -8.0)
good.fill_diagonal_(8.0)
bad = good[:, torch.tensor([1, 0, 3, 2])]

assert torch.allclose(clip_loss(zero), torch.tensor(math.log(B)))
assert torch.allclose(siglip_loss(zero), torch.tensor(B * math.log(2)))
assert clip_loss(good) < clip_loss(zero) < clip_loss(bad)
assert siglip_loss(good) < siglip_loss(zero) < siglip_loss(bad)
assert clip_loss(torch.zeros(1, 1)).item() == 0.0
assert torch.allclose(siglip_loss(torch.zeros(1, 1)),
                      torch.tensor(math.log(2)))

z = 2 * torch.eye(B) - 1
assert torch.allclose(siglip_loss(good), F.softplus(-z * good).sum() / B)
for name, fn in [('CLIP', clip_loss), ('SigLIP', siglip_loss)]:
    print(name, {key: fn(value).item()
                 for key, value in [('zero', zero), ('good', good), ('bad', bad)]})
```

Ожидаемые точные ориентиры: при $B=4$ нулевые логиты дают $\ln4\approx1.3863$ для CLIP и $4\ln2\approx2.7726$ для принятой нормировки SigLIP. При $B=1$ у CLIP нет конкурирующего отрицательного примера, и потеря равна нулю. У SigLIP остаётся положительная бинарная пара. Численные значения двух разных функций потерь **не являются общей шкалой качества моделей**.

### Смещение и градиенты

При одной положительной паре на строку доля положительных среди всех пар равна $1/B$. Если все логиты одинаковы и $B>1$, минимум бинарной цели достигается при

$$
b^*=\log\frac{1/B}{1-1/B}=-\log(B-1).
$$

Это решение для постоянного предсказателя при данном соотношении пар, а не утверждение о конечном смещении обученной модели.

```python
bias = torch.tensor(-math.log(B - 1), requires_grad=True)
constant_logits = bias.expand(B, B)
loss = siglip_loss(constant_logits)
loss.backward()
assert abs(bias.grad.item()) < 1e-5
print('optimal constant bias:', bias.item(), 'gradient:', bias.grad.item())

s = torch.zeros(B, B, requires_grad=True)
siglip_loss(s).backward()
assert (s.grad.diag() < 0).all()
off_diagonal = ~torch.eye(B, dtype=torch.bool)
assert (s.grad[off_diagonal] > 0).all()
print('SigLIP gradient at zero:\n', s.grad)
```

**Задание:** самостоятельно получить градиент CLIP по матрице логитов и сравнить с SigLIP. При симметричной цели не забыть, что каждая ячейка участвует и в нормировке строки, и в нормировке столбца.

### Ложные отрицательные

Если двум изображениям соответствует одно описание `a photo of a dog`, диагональная разметка заставляет считать одну семантически корректную пару отрицательной. Покажите такую ситуацию на двух изображениях одного класса из CIFAR-10. Объясните, почему это проблема построения обучающих пар, а не ошибка функции `cross_entropy`.

Для бинарной цели можно задать несколько положительных пар маской. Следующий код иллюстрирует именно **классовую** релевантность; он не утверждает, что любое описание подходит любому изображению своего класса.

```python
labels_demo = torch.tensor([0, 0, 1, 1])
positive_mask = labels_demo[:, None].eq(labels_demo[None, :]).float()
logits_demo = torch.zeros(4, 4, requires_grad=True)
loss_multi = F.binary_cross_entropy_with_logits(
    logits_demo, positive_mask, reduction='sum') / len(labels_demo)
loss_multi.backward()
assert logits_demo.grad[0, 1] < 0  # вторая пара того же класса положительна
print('multi-positive loss:', loss_multi.item())
```

**Вопрос:** почему при изменении числа положительных и отрицательных меняются и оптимальное постоянное смещение, и интерпретация вероятности? Как изменится масштаб градиента при переходе от `sum()/B` к `mean()`?

---

## S5. Дополнительное задание: линейная голова на замороженных признаках

Эта часть продолжает семинар 2 и выполняется дома либо при наличии времени. Обучаем только линейный классификатор поверх CLIP; для SigLIP повторяем тот же протокол отдельно. Энкодер остаётся замороженным. Для обучения берём новые изображения из `train`, не пересекающиеся с проверочной выборкой, а не используем проверочные признаки как обучающие.

```python
TRAIN_PER_CLASS = 50
rng = np.random.default_rng(SEED + 2)
all_train_labels = np.asarray(train_ds.targets)
allowed = np.ones(len(train_ds), dtype=bool)
allowed[val_ids] = False
probe_ids = np.concatenate([
    rng.choice(np.flatnonzero((all_train_labels == c) & allowed),
               TRAIN_PER_CLASS, replace=False)
    for c in range(len(classes))
])
rng.shuffle(probe_ids)
assert not set(probe_ids.tolist()) & set(val_ids.tolist())
probe_images = [train_ds[int(i)][0].convert('RGB') for i in probe_ids]
y_probe = torch.tensor(all_train_labels[probe_ids], dtype=torch.long)

kind = 'CLIP'  # затем повторить для SigLIP
model_id, model_class = MODEL_SPECS[kind]
processor = AutoProcessor.from_pretrained(model_id)
model = model_class.from_pretrained(model_id).to(device).eval()
model.requires_grad_(False)
x_probe = encode_images(model, processor, probe_images)
del model, processor
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Маленькую голову обучаем на CPU; все признаки уже расположены на CPU.
torch.manual_seed(SEED)
head = torch.nn.Linear(x_probe.shape[1], len(classes))
optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
best_val = -1.0
best_state = None
history = []
for epoch in range(100):
    head.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(head(x_probe), y_probe)
    loss.backward()
    optimizer.step()
    head.eval()
    with torch.no_grad():
        val_acc = (head(cache[kind]['val']).argmax(-1) == y_val).float().mean().item()
    history.append({'epoch': epoch + 1, 'train_loss': loss.item(),
                    'val_accuracy': val_acc})
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

head.load_state_dict(best_state)
with torch.no_grad():
    pred = head(cache[kind]['test']).argmax(-1).numpy()
print('probe accuracy:', accuracy_score(y_test.numpy(), pred))
print('probe macro-F1:', f1_score(y_test.numpy(), pred, average='macro', zero_division=0))
pd.DataFrame(history).to_csv(OUT / f'{kind}_probe_history.csv', index=False)
np.save(OUT / 'probe_train_ids.npy', probe_ids)
torch.save(best_state, OUT / f'{kind}_probe_head.pt')
```

Сравните голову с текстовым классификатором по одним и тем же тестовым изображениям. Укажите, что линейная голова использовала целевую разметку, а базовый текстовый классификатор — нет. Подбор шаблона по размеченной проверочной выборке тоже нужно явно отметить: строгий режим без целевой разметки соответствует заранее фиксированному шаблону `photo`. Прирост линейной головы не гарантируется, особенно на маленькой выборке.

---

## Итоговый отчёт и критерии готовности

Сдать ноутбук с выполненными ячейками и короткий отчёт в Markdown. Достаточно одного репозитория или архива с ноутбуком, отчётом и каталогом `results_seminar3`. Полные веса моделей сдавать не требуется.

В отчёте должны быть:

1. Окружение, устройство, идентификаторы и ревизии моделей, размеры выборок и сохранённые индексы.
2. Таблица accuracy и macro-F1 на проверочной выборке для четырёх вариантов текстовых прототипов.
3. Итоговая тестовая таблица для фиксированного и выбранного на проверке шаблонов, две матрицы ошибок.
4. Precision@5, Recall@5 и Hit@5 с явным определением релевантности; галереи результатов поиска и минимум три разобранные ошибки.
5. Результаты численных проверок функций потерь и объяснение случаев $B=1$, нулевых логитов и ложных отрицательных.
6. Краткий вывод: что зависит от формулировки текста, что — от архитектуры и предобучения, а какие выводы этот эксперимент вообще не позволяет сделать.

**Минимальный результат:** обе модели работают; признаки и ручные логиты проверены; классификация и поиск измерены; код функций потерь проходит приведённые проверки. Если доступна только одна модель, работу можно продолжить с ней, но сравнение двух моделей остаётся невыполненной частью задания.

**Полный результат:** выполнены все основные этапы S0–S4, выдержан протокол разделения данных, результаты сопровождаются разбором ошибок. Дополнительный S5 и опыт с собственными подписями расширяют работу, но не заменяют обязательные проверки.

### Вопросы для обсуждения в конце пары

Почему два независимо обученных энкодера нельзя просто соединить через косинусное сходство? Что именно делает текстовый вектор весом классификатора? Чем усреднение нормированных текстовых векторов отличается от усреднения вероятностей? Почему успех поиска по названию класса не доказывает понимания пространственных отношений? Почему размер пакета важен для обучения CLIP, но при поиске признаки индекса можно вычислять маленькими пакетами? Какая проверка понадобилась бы для порога отказа от ответа?

## Материалы

- Лекция 3 «Совместные представления изображения и текста: CLIP и SigLIP», разделы 1–4.
- [CLIP: оригинальная статья](https://arxiv.org/abs/2103.00020).
- [SigLIP: оригинальная статья](https://arxiv.org/abs/2303.15343).
- [Документация CLIP, Transformers 4.46.3](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/clip).
- [Документация SigLIP, Transformers 4.46.3](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/siglip).
- [Контрольная точка CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [контрольная точка SigLIP](https://huggingface.co/google/siglip-base-patch16-224).
- [CIFAR-10: описание набора данных](https://www.cs.toronto.edu/~kriz/cifar.html).
