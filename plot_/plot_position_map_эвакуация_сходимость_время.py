import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# === Параметры ===
ALGORITHMS = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
]

CSV_TEMPLATE = "metrics_1000_{}.csv"
TOTAL_AGENTS = 440
WINDOW = 50
OUTPUT = "position_map.png"

# === Сбор характеристик
summary = []

for algo in ALGORITHMS:
    path = Path(CSV_TEMPLATE.format(algo))
    if not path.exists():
        print(f"⚠️ Файл не найден: {path}")
        continue

    df = pd.read_csv(path)
    if "evacuated" not in df.columns or "episode" not in df.columns:
        print(f"⚠️ Пропущено: {algo}, нет 'evacuated' или 'episode'")
        continue

    evac = df["evacuated"].rolling(WINDOW, min_periods=1).mean()
    evac_pct = evac / TOTAL_AGENTS * 100

    # ⚠️ Только последние 500 эпизодов
    evac_pct_last500 = evac_pct[-500:]
    mean_evac = evac_pct_last500.mean()

    max_evac = evac_pct.max()
    threshold = max_evac * 0.95
    convergence_ep = next((i for i, val in enumerate(evac_pct) if val >= threshold), len(evac_pct))

    duration_total = df["duration"].sum() if "duration" in df.columns else 0

    summary.append({
        "algorithm": algo,
        "evac_percent": mean_evac,
        "convergence_ep": convergence_ep,
        "duration_total": duration_total
    })

# === Построение карты
summary_df = pd.DataFrame(summary)

# Нормируем размер точек
max_duration = summary_df["duration_total"].max()
summary_df["size"] = (summary_df["duration_total"] / max_duration) * 1000 + 200  # базовый размер

# Цвета
colors = plt.cm.Set2(np.linspace(0, 1, len(summary_df)))

plt.figure(figsize=(12, 7))
for i, row in summary_df.iterrows():
    plt.scatter(
        row["evac_percent"],
        row["convergence_ep"],
        s=row["size"],
        color=colors[i],
        alpha=0.85,
        edgecolor='black'
    )
    plt.text(
        row["evac_percent"] + 0.5,
        row["convergence_ep"],
        f"{row['algorithm']}\n{int(row['duration_total'])}s",
        fontsize=9
    )

# === Статистический тест: сравнение всех алгоритмов попарно
from itertools import combinations
from scipy import stats

print("\n=== Статистическая значимость (Mann–Whitney U-тест, последние 500 эпизодов) ===")

evac_all = {}

for algo in ALGORITHMS:
    path = Path(CSV_TEMPLATE.format(algo))
    if not path.exists():
        continue

    df = pd.read_csv(path)
    evac_pct = df["evacuated"].rolling(WINDOW, min_periods=1).mean() / TOTAL_AGENTS * 100
    evac_all[algo] = evac_pct[-500:].values

# Попарные сравнения
for a1, a2 in combinations(ALGORITHMS, 2):
    if a1 not in evac_all or a2 not in evac_all:
        continue

    u, p = stats.mannwhitneyu(evac_all[a1], evac_all[a2], alternative='two-sided')
    significance = "✅" if p < 0.05 else "—"
    print(f"{a1:24s} vs {a2:24s} | p = {p:.4f} {significance}")

    mean1 = evac_all[a1].mean()
    mean2 = evac_all[a2].mean()
    diff = abs(mean1 - mean2)
    print(f"{a1:24s} vs {a2:24s} | p = {p:.5f} {significance} | Δmean = {diff:.2f} %")



plt.xlabel("Средняя эвакуация (% от всех агентов, последние 500 эп.)", fontsize=12)
plt.ylabel("Эпизод выхода на 90% от max эвакуации", fontsize=12)
plt.title("Позиционная карта алгоритмов (эвакуация × сходимость × длительность)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT)
plt.show()

