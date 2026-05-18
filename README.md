# VegVision

Минимальный дипломный MVP для обучения и сравнения YOLO-моделей на задаче обнаружения болезней овощных культур в теплице.

## Структура проекта

- `data/datasets/tomatoes_v1/` — датасет томатов (болезни) в формате Ultralytics YOLO object detection.
- `configs/` — общие параметры эксперимента, конфиги моделей и шаблон датасета.
  - `configs/experiment.yaml` — основной конфиг для полного обучения (100 эпох).
  - `configs/experiment_pilot.yaml` — конфиг для пилот-тестирования (3 эпохи).
  - `configs/models/` — конфиги для каждой модели (yolo11n, yolo11l, yolo26n, yolo26l).
- `src/train.py` — запуск обучения одной модели по YAML-конфигам.
- `src/validate.py` — валидация обученной модели и сохранение метрик.
- `src/compare_models.py` — сбор метрик и формирование сравнительных таблиц (edge, cloud, cascade).
- `src/utils/` — вспомогательные функции для путей, YAML, логов и метрик.
- `runs/` — артефакты обучения Ultralytics.
- `results/metrics/` — JSON с метриками валидации.
- `results/comparisons/` — сравнительные таблицы (CSV/MD) по ролям моделей.
- `results/logs/` — summary-файлы обучающих запусков.

## Датасет

Датасет **tomatoes_v1** находится в `data/datasets/tomatoes_v1/` и содержит:

- 9 классов болезней/состояний листьев томатов
- Стандартная структура: `train/`, `valid/`, `test/` с подпапками `images/` и `labels/`
- `data.yaml` — конфиг датасета с описанием классов и путей

## Быстрый старт: Пилот-обучение

Для проверки конфигурации и обучаемости датасета запусти пилот-обучение (3 эпохи):

```powershell
& ".venv\Scripts\python.exe" run_pilot_training.py
```

Будут обучены все 4 модели (YOLO11n/11l, YOLO26n/26l) с минимальным количеством эпох.

## Полное обучение

После успешного пилот-тестирования запусти полное обучение (100 эпох):

```powershell
& ".venv\Scripts\python.exe" src\train.py --experiment-config configs\experiment.yaml --model-config configs\models\yolo11n.yaml
```

Для каждой модели:

```powershell
& ".venv\Scripts\python.exe" src\train.py --experiment-config configs\experiment.yaml --model-config configs\models\yolo11l.yaml
& ".venv\Scripts\python.exe" src\train.py --experiment-config configs\experiment.yaml --model-config configs\models\yolo26n.yaml
& ".venv\Scripts\python.exe" src\train.py --experiment-config configs\experiment.yaml --model-config configs\models\yolo26l.yaml
```

## Валидация

После обучения запусти валидацию для всех моделей:

```powershell
& ".venv\Scripts\python.exe" run_validation.py
```

Или валидируй отдельную модель:

```powershell
& ".venv\Scripts\python.exe" src\validate.py --experiment-config configs\experiment.yaml --model-config configs\models\yolo11n.yaml --run-dir runs\train\yolo11n_20260518_135335
```

Метрики сохраняются в `results/metrics/` в JSON-формате.

## Сравнение моделей

Для создания сравнительных таблиц по ролям (edge/cloud) и каскадных сравнений:

```powershell
& ".venv\Scripts\python.exe" run_comparison.py
```

Будут созданы:

- `results/comparisons/edge_models_comparison.csv/.md` — сравнение edge-моделей (YOLO11n, YOLO26n)
- `results/comparisons/cloud_models_comparison.csv/.md` — сравнение cloud-моделей (YOLO11l, YOLO26l)
- `results/comparisons/cascade_comparison.csv/.md` — каскадные пары (edge+cloud по семействам)

## Роли моделей

- **Edge**: YOLO11n, YOLO26n — легкие модели для фильтрации на устройстве
  - Приоритет: recall (высокая чувствительность), скорость, размер модели
  - Цель: не пропустить потенциальные проблемы
  
- **Cloud**: YOLO11l, YOLO26l — тяжелые модели для уточнения в облаке
  - Приоритет: precision (высокая точность), mAP50/mAP50-95, скорость
  - Цель: минимизировать ложные срабатывания

- **Каскад**: edge → cloud — двухступенчатая фильтрация
  - Edge отсеивает негативные примеры
  - Cloud подтверждает и классифицирует позитивные

## Примечания

- Все пути в конфигах разрешаются относительно корня проекта (`vegvision-yolo/`)
- Переменные окружения `ULTRALYTICS_SKIP_CHECKS=1` отключают проверку обновлений (требует интернета)
- Веса моделей автоматически загружаются при первом запуске обучения

## Как сравнивать модели

```powershell
& ".venv\Scripts\python.exe" src\compare_models.py
```

Скрипт соберёт все JSON-метрики из `results/metrics/` и создаст:

- `results/comparisons/model_comparison.csv`
- `results/comparisons/model_comparison.md`

## Какие метрики сравниваются

- precision
- recall
- mAP@50
- mAP@50-95
- скорость инференса
- размер модели

## Что менять под свои эксперименты

- `configs/experiment.yaml` — общие параметры обучения и сравнения.
- `configs/models/*.yaml` — источник модели (`model_source`) и её роль.
- `configs/datasets/*.yaml` — путь к локальному датасету и список классов.

## Примечание по YOLO26

Если готовых весов нет, в `configs/models/yolo26*.yaml` можно заменить `model_source.value` или `fallback_value` на другой совместимый источник — `.pt`, `.yaml` или alias, который понимает Ultralytics.
