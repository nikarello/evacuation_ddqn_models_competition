import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# 🔽 ДОБАВЬ ЭТО СРАЗУ ПОСЛЕ
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})


# Папка с метриками
metrics_dir = Path("liders_check")

# Алгоритмы
algorithms = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN"
]

# Лидеры
leader_counts = list(range(0, 21))  # от 0 до 20

# Параметры
TOTAL_AGENTS = 440
SMOOTH_WINDOW = 5

# Цвета и подписи статистик
stats_info = {
    'mean':   ('red',    'mean'),
    'median': ('orange', 'median'),
    'min':    ('blue',   'min'),
    'max':    ('green',  'max'),
}

# === Один график на алгоритм ===
for algo in algorithms:
    heat_data = []

    for n in leader_counts:
        csv_path = metrics_dir / f"leaders_{n}_{algo}.csv"
        if not csv_path.exists():
            row = [np.nan] * 50
        else:
            df = pd.read_csv(csv_path)
            evacuated = df["evacuated"].values[:50]
            perc = (evacuated / TOTAL_AGENTS) * 100
            perc = pd.Series(perc).rolling(window=SMOOTH_WINDOW, min_periods=1).mean().values
            row = perc
        heat_data.append(row)

    heat_data = np.array(heat_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"{algo} — эвакуация в зависимости от числа лидеров", fontsize=18)

    # === Тепловая карта ===
    sns.heatmap(heat_data, ax=ax1, cmap="YlGnBu", cbar=True, vmin=0, vmax=100)
    ax1.set_title(f"Heatmap эвакуации (%)", fontsize=14)
    ax1.set_ylabel("Число лидеров")
    ax1.set_xlabel("Эпизод")
    ax1.set_yticks(np.arange(len(leader_counts)) + 0.5)
    ax1.set_yticklabels([f"{n}" for n in leader_counts], rotation=0)

    stat_coords = {key: ([], []) for key in stats_info}
    for row_idx, row in enumerate(heat_data):
        vals = {
            'mean':   np.nanmean(row),
            'median': np.nanmedian(row),
            'min':    np.nanmin(row),
            'max':    np.nanmax(row),
        }
        for stat, (color, _) in stats_info.items():
            val = vals[stat]
            if not np.isnan(val):
                col_idx = np.abs(row - val).argmin()
                ax1.plot(col_idx + 0.5, row_idx + 0.5, 'o', color=color, markersize=6)
                stat_coords[stat][0].append(col_idx + 0.5)
                stat_coords[stat][1].append(row_idx + 0.5)

    for stat, (color, label) in stats_info.items():
        x_vals, y_vals = stat_coords[stat]
        if len(x_vals) >= 2:
            ax1.plot(x_vals, y_vals, color=color, linewidth=1.0, alpha=0.6, label=label)

    ax1.set_xticks(np.arange(0, heat_data.shape[1], 5) + 0.5)
    ax1.set_xticklabels([str(i) for i in range(0, heat_data.shape[1], 5)])
    ax1.legend(loc="upper left", fontsize=9)

    # === Траектории ===
    for n, row in zip(leader_counts, heat_data):
        if not np.isnan(row).all():
            ax2.plot(np.arange(len(row)), row, label=f"{n} лидеров", alpha=0.5)

    ax2.set_title(f"Траектории эвакуации (%)", fontsize=14)
    ax2.set_xlabel("Эпизод")
    ax2.set_ylabel("Процент эвакуированных")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend(fontsize=7, ncol=2, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # === Сохраняем график ===
    save_path = Path(f"evacuation_{algo}.png")
    plt.savefig(save_path)
    print(f"✅ Сохранено: {save_path}")
    plt.close()
