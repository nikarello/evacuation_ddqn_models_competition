# evacuation_ddqn_models_competition
Сравнение Dueling DDQN, QRDQN, Prioritized и Noisy DQN на задаче эвакуации агентов

---

## 🧠 СТРУКТУРА ПРОЕКТА

### 📌 Основные компоненты:
- `main.py` — основной файл запуска обучения
- `environment.py` — симулятор среды с агентами, огнём и выходами
- `core.py` — базовые функции, награды, обновления, буфер
- `algorithms/` — реализация всех RL-алгоритмов:
  - `DuelingDDQNTrainer`, `QRDQNTrainer`, `NoisyDuelingDDQNTrainer`, `DuelingDDQNPrioritizedTrainer`
- `models/` — CNN и NoisyCNN архитектуры
- `core_modules/` — приоритизированный буфер на GPU
- `plot_/` — визуализация и анализ результатов

### 🧩 Взаимодействие классов:

![class_interaction_graph](https://github.com/user-attachments/assets/d55ef0ab-d875-45bc-bee1-979b03cab498)

---

## 🚀 Запуск проекта

### 📄 Описание
Этот проект моделирует эвакуацию агентов в среде с огнём с использованием различных алгоритмов обучения с подкреплением (Dueling DDQN, QR-DQN и др.). Поддерживается как последовательное, так и параллельное обучение, а также специальные режимы для проверки гипотез.

---

### ⚙️ Основной запуск

<pre>
```bash
python main.py --config config.json
```
</pre>

- Загружает конфигурацию
- Создаёт среду эвакуации
- Инициализирует выбранный алгоритм (`ALGORITHM`)
- Запускает обучение: `trainer.train()`

---

### 🔁 Пакетный запуск всех алгоритмов

<pre>
```bash
python run_all.py
```
</pre>

- Последовательно обучает:
  - `DuelingDDQN`
  - `QRDQN`
  - `DuelingDDQNPrioritized`
  - `NoisyDuelingDDQN`
- Для каждого:
  - Изменяет параметр `ALGORITHM` в `config.json`
  - Запускает `main.py`
  - Сохраняет метрики, графики, видео

---

### ⚡ Параллельный запуск всех алгоритмов

<pre>
```bash
python run_all_parallel.py
```
</pre>

- Запускает все алгоритмы одновременно
- Использует несколько процессов
- Подходит для многопроцессорных систем и ускоренной отладки

---

### 🧠 Исследование гипотез

#### 📍 С лидером

<pre>
```bash
python run_leader_study.py
```
</pre>

- Активирует агентов-лидеров (`NUM_LEADERS > 0`)
- Проверяет влияние информированного агента на общую эффективность эвакуации

#### 📍 Задымление

<pre>
```bash
python run_smoke_study.py
```
</pre>

- Проверяет влияние VIEW_SIZE на общую эффективность эвакуации

---

---

### 📊 Результаты

- Все метрики сохраняются в `.csv` в корне проекта
- Видеоэпизоды — в `videos/`
- Графики — в `figures/` (если включены)
- Поддерживаются скрипты из папки `plot_/` для анализа:

<pre>
```bash
python plot_/plot_all_metrics.py
```
</pre>

---

### 📁 Поддерживаемые алгоритмы

- `DuelingDDQNTrainer`
- `QRDQNTrainer`
- `DuelingDDQNPrioritizedTrainer`
- `NoisyDuelingDDQNTrainer`

Каждый реализован в `algorithms/` и наследует интерфейс `BaseTrainer`.

---

