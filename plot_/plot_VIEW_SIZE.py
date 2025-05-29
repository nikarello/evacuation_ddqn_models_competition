import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# === Настройки ===
FOLDER = Path("smoke_check")
METRIC = "evacuated"
ROLLING = 20
PALETTE = sns.color_palette("husl", 10)
csv_files = list(FOLDER.glob("smoke_*.csv"))

TOTAL_AGENTS = 480  # 48 × 10 сред

records = []

# === Чтение CSV-файлов ===
for file in csv_files:
    match = re.match(r"smoke_(\d+)_(.+)\.csv", file.name)
    if not match:
        continue
    view_size = int(match.group(1))
    algo = match.group(2)

    try:
        df = pd.read_csv(file)
        df["view_size"] = view_size
        df["algorithm"] = algo
        df["evacuated_pct"] = df["evacuated"] / TOTAL_AGENTS * 100
        records.append(df)
    except Exception as e:
        print(f"⚠️ Ошибка при чтении {file.name}: {e}")

# === Объединение ===
df_all = pd.concat(records, ignore_index=True)

# === Подготовка графиков ===
algorithms = sorted(df_all["algorithm"].unique())
view_sizes = sorted(df_all["view_size"].unique())

sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
axes = axes.flatten()

for idx, algo in enumerate(algorithms):
    ax = axes[idx]
    df_algo = df_all[df_all["algorithm"] == algo]

    for i, v in enumerate(view_sizes):
        df_sub = df_algo[df_algo["view_size"] == v]
        grouped = df_sub.groupby("episode")["evacuated_pct"].mean().reset_index()
        rolled = grouped["evacuated_pct"].rolling(window=ROLLING, min_periods=1).mean()
        ax.plot(grouped["episode"], rolled, label=f"VIEW_SIZE = {v}", color=PALETTE[i % len(PALETTE)])

    ax.set_title(algo, fontsize=14)
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Эвакуировано (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True)

fig.suptitle("Сравнение алгоритмов — процент эвакуированных по VIEW_SIZE", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = FOLDER / "evac_pct_by_viewsize_per_algo.png"
plt.savefig(out_path, dpi=300)
print(f"✅ График сохранён: {out_path}")
plt.show()
