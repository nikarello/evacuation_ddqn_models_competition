import pandas as pd
from scipy.stats import kruskal
from pathlib import Path
import re

# === ПАРАМЕТРЫ ===
FOLDER = Path("liders_check")
ALGORITHMS = ["NoisyDuelingDDQN", "DuelingDDQN", "DuelingDDQNPrioritized", "QRDQN"]
TARGET_METRIC = "evacuated"
MAX_EVAC = 48  # общее число агентов, для нормировки в %

# === РЕЗУЛЬТАТЫ ===
print("=== 🧪 Kruskal–Wallis H-тест по evacuated_ratio ===\n")
for algo in ALGORITHMS:
    grouped = {}
    files = list(FOLDER.glob(f"leaders_*_{algo}.csv"))

    for file in files:
        match = re.match(r"leaders_(\d+)_", file.name)
        if not match:
            continue
        n_leaders = int(match.group(1))
        df = pd.read_csv(file)
        if TARGET_METRIC not in df.columns:
            continue
        values = df[TARGET_METRIC] / MAX_EVAC * 100
        grouped.setdefault(n_leaders, []).extend(values.dropna().tolist())

    sorted_groups = sorted(grouped.items())
    if len(sorted_groups) < 2:
        print(f"⚠️ Недостаточно данных для {algo}")
        continue

    group_labels = [f"{k} лидеров" for k, _ in sorted_groups]
    group_values = [v for _, v in sorted_groups]

    stat, p = kruskal(*group_values)
    print(f"{algo}: H = {stat:.4f}, p = {p:.5f} {'✅ значимо' if p < 0.05 else '❌ незначимо'}")
