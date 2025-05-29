import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ALGORITHMS = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
]

CSV_TEMPLATE = "metrics_1000_{}.csv"
PERCENT_METRICS = {"evac_min": 48, "evac_max": 48}
OUTPUT_DIR = Path("plots_metrics")
OUTPUT_DIR.mkdir(exist_ok=True)

summary = []
for algo in ALGORITHMS:
    path = Path(CSV_TEMPLATE.format(algo))
    if not path.exists():
        continue

    df = pd.read_csv(path)
    if not {"evac_min", "evac_max", "avg_hp"}.issubset(df.columns):
        print(f"⚠️ Пропуск {algo}: отсутствуют нужные колонки")
        continue

    evac_max = df["evac_max"].rolling(50, min_periods=1).mean().iloc[-500:].mean() / PERCENT_METRICS["evac_max"] * 100
    evac_min = df["evac_min"].rolling(50, min_periods=1).mean().iloc[-500:].mean() / PERCENT_METRICS["evac_min"] * 100
    avg_hp = df["avg_hp"].rolling(50, min_periods=1).mean().iloc[-500:].mean()

    summary.append({
        "algorithm": algo,
        "evac_max": evac_max,
        "evac_min": evac_min,
        "avg_hp": avg_hp,
    })

df_summary = pd.DataFrame(summary)

# Масштабируем размер шаров
min_size, max_size = 200, 1200
hp_min, hp_max = df_summary["avg_hp"].min(), df_summary["avg_hp"].max()
sizes = min_size + (df_summary["avg_hp"] - hp_min) / (hp_max - hp_min + 1e-6) * (max_size - min_size)

# Построение графика
plt.figure(figsize=(10, 8))
colors = sns.color_palette("Set2", len(df_summary))

for i, row in df_summary.iterrows():
    plt.scatter(row["evac_max"], row["evac_min"], s=sizes[i], color=colors[i], alpha=0.8)
    label = f"{row['algorithm']} ({row['avg_hp']:.1f} HP)"
    plt.text(row["evac_max"] - 5, row["evac_min"], label, fontsize=10, ha='right', va='center')

plt.xlabel("Максимум эвакуации малых групп (%)")
plt.ylabel("Минимум эвакуации малых групп (%)")
plt.title("Размах эвакуации малых групп\nРазмер - Среднее HP (последние 500 эпизодов)")
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()

# Сохраняем изображение
plt.savefig(OUTPUT_DIR / "scatter_evac_range.png")
print("✅ Сохранено: scatter_evac_range.png")
plt.close()
