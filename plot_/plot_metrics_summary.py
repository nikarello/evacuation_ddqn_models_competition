# plot_metrics_summary.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки
METRIC_FILES = [
    "metrics_DuelingDDQN.csv",
    "metrics_QRDQN.csv",
    "metrics_DuelingDDQNPrioritized.csv",
    "metrics_NoisyDuelingDDQN.csv"
]

EXCLUDE_COLUMNS = {"episode", "lr"}
ROLLING_WINDOW = 10

# Палитра
PALETTE = {
    "DuelingDDQN": "darkred",
    "QRDQN": "orange",
    "DuelingDDQNPrioritized": "orchid",
    "NoisyDuelingDDQN": "deeppink"
}

for file in METRIC_FILES:
    if not Path(file).exists():
        raise FileNotFoundError(f"Файл не найден: {file}")

# Загрузка и объединение
all_dfs = []
for file in METRIC_FILES:
    algo = Path(file).stem.replace("metrics_", "")
    df = pd.read_csv(file)
    df["algorithm"] = algo
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# Получаем список метрик
metrics = [col for col in df_all.columns if col not in EXCLUDE_COLUMNS and col not in ("algorithm")]

# Построение графиков
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=(len(metrics) + 1) // 2, ncols=2, figsize=(18, 24))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for algo in df_all["algorithm"].unique():
        df_algo = df_all[df_all["algorithm"] == algo].sort_values("episode")
        rolled = df_algo[metric].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ax.plot(df_algo["episode"], rolled, label=algo, color=PALETTE.get(algo))

    ax.set_title(metric, fontsize=14)
    ax.set_xlabel("Эпизод")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(fontsize=9)

# Удалим пустые оси, если они есть
for j in range(len(metrics), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
out_path = "metrics_comparison_all_algorithms.png"
plt.savefig(out_path, dpi=300)
print(f"✅ График сохранён: {out_path}")
