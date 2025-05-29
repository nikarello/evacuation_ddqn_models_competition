# plot_per_leader_variant.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Настройки
FOLDER = Path("liders_check")
METRICS = ["evacuated", "avg_hp", "avg_steps", "reward"]
PALETTE = sns.color_palette("husl", 10)  # до 10 цветов
ROLLING = 5  # скользящее среднее

# Читаем CSV-файлы
records = []
csv_files = list(FOLDER.glob("leaders_*.csv"))

for file in csv_files:
    match = re.match(r"leaders_(\d+)_(.+)\.csv", file.name)
    if not match:
        continue
    n_leaders = int(match.group(1))
    algo = match.group(2)

    try:
        df = pd.read_csv(file)
        df["leaders"] = n_leaders
        df["algorithm"] = algo
        records.append(df)
    except Exception as e:
        print(f"❌ Ошибка при чтении {file.name}: {e}")

# Объединяем
df_all = pd.concat(records, ignore_index=True)

# Уникальные алгоритмы и числа лидеров
algorithms = sorted(df_all["algorithm"].unique())
leader_values = sorted(df_all["leaders"].unique())

# Отрисовка
for algo in algorithms:
    df_algo = df_all[df_all["algorithm"] == algo]

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle(f"{algo} — влияние числа лидеров", fontsize=18)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for i, n_leaders in enumerate(leader_values):
            df_sub = df_algo[df_algo["leaders"] == n_leaders]
            df_sub = df_sub.sort_values("episode")
            series = df_sub[metric].rolling(window=ROLLING, min_periods=1).mean()
            ax.plot(df_sub["episode"], series, label=f"{n_leaders} лидеров", color=PALETTE[i % len(PALETTE)])
        
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel("Эпизод")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FOLDER / f"plot_leader_variants_{algo}.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Сохранён график: {out_path}")
