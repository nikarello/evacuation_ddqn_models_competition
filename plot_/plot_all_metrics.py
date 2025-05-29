import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# === Алгоритмы и метрики ===
ALGORITHMS = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
]

METRICS = [
    "reward",
    "evacuated",
    "avg_steps",
    "avg_hp",
    "loss",
    "duration",
    "evac_min",
    "evac_max"
]

# Метрики в процентах
PERCENT_METRICS = {
    "evacuated": 480,
    "evac_min": 48,
    "evac_max": 48,
}

SMOOTH_WINDOW = 50
CSV_TEMPLATE = "metrics_1000_{}.csv"
OUTPUT_DIR = Path("plots_metrics")
OUTPUT_DIR.mkdir(exist_ok=True)

# === График значения метрики ===
def plot_metric(metric):
    plt.figure(figsize=(12, 6))
    for algo in ALGORITHMS:
        path = Path(CSV_TEMPLATE.format(algo))
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue

        y = df[metric]
        if metric in PERCENT_METRICS:
            y = (y / PERCENT_METRICS[metric]) * 100
        smooth = y.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        plt.plot(df["episode"], smooth, label=algo)

    plt.title(f"{metric} (окно {SMOOTH_WINDOW})")
    plt.xlabel("Эпизод")
    plt.ylabel("%" if metric in PERCENT_METRICS else metric)
    if metric == "reward":
        plt.yscale("symlog")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"plot_{metric}.png")
    plt.close()

# === График сходимости (|Δ| по сглаженному значению) ===
def plot_convergence(metric):
    plt.figure(figsize=(12, 6))
    for algo in ALGORITHMS:
        path = Path(CSV_TEMPLATE.format(algo))
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue

        y = df[metric]
        if metric in PERCENT_METRICS:
            y = (y / PERCENT_METRICS[metric]) * 100
        smooth = y.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        delta = smooth.diff().abs().rolling(window=50, min_periods=1).mean()

        plt.plot(df["episode"], delta, label=algo)

    plt.title(f"Сходимость: |Δ {metric}| (log-scale)")
    plt.xlabel("Эпизод")
    plt.ylabel("Δ (абсолютное изменение)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"convergence_{metric}.png")
    plt.close()

# === Запуск
if __name__ == "__main__":
    for metric in METRICS:
        plot_metric(metric)
        plot_convergence(metric)
        print(f"✅ Построены: plot_{metric}.png и convergence_{metric}.png")
