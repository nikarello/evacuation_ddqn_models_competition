import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === Параметры ===
METRICS = ['evacuated', 'avg_steps', 'duration', 'evac_min', 'evac_max']
FOLDERS = {
    'simple': Path('метрики 1000/дефолт'),
    'fire spawn fix': Path('метрики 1000/фикс спавна огня'),
}
ALGORITHMS = [
    "DuelingDDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
    "QRDQN"
]
N_LAST_EPISODES = 100

# === Сбор относительных изменений ===
heatmap_data = {metric: {} for metric in METRICS}

for algo in ALGORITHMS:
    df_simple = pd.read_csv(FOLDERS['simple'] / f"metrics_{algo}.csv").tail(N_LAST_EPISODES)
    df_fixed  = pd.read_csv(FOLDERS['fire spawn fix'] / f"metrics_{algo}.csv").tail(N_LAST_EPISODES)

    for metric in METRICS:
        val_simple = df_simple[metric].mean()
        val_fixed  = df_fixed[metric].mean()

        if val_simple != 0:
            delta_pct = (val_fixed - val_simple) / val_simple * 100
        else:
            delta_pct = 0

        heatmap_data[metric][algo] = delta_pct

# === Построение 5 тепловых карт в одной фигуре ===
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 12), constrained_layout=True)

for idx, metric in enumerate(METRICS):
    df_metric = pd.DataFrame({metric: heatmap_data[metric]}).T
    ax = axes[idx]
    sns.heatmap(df_metric.T, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                cbar_kws={'label': '%'}, ax=ax)
    ax.set_title(f"{metric}", fontsize=12)
    ax.set_xlabel("Изменение (%)")
    ax.set_ylabel("")

fig.suptitle("📊 Относительное изменение метрик (fix vs default)", fontsize=16)
plt.savefig("heatmap_relative_subplots.png")
plt.show()
