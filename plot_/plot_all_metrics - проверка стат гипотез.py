import pandas as pd
from scipy.stats import mannwhitneyu
from pathlib import Path

# === Настройки ===
FOLDERS = {
    'base': Path('метрики 1000/дефолт'),
    'fix': Path('метрики 1000/фикс спавна огня'),
}
ALGORITHMS = [
    "DuelingDDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
    "QRDQN"
]
TOTAL_AGENTS = 1200
N_LAST = 100

print("=== 🔥 H1: Проверка влияния запрета спавна огня у выхода ===")

for algo in ALGORITHMS:
    try:
        df_base = pd.read_csv(FOLDERS['base'] / f"metrics_{algo}.csv")
        df_fix = pd.read_csv(FOLDERS['fix'] / f"metrics_{algo}.csv")
    except FileNotFoundError:
        print(f"⚠️ Пропущено: нет файлов для {algo}")
        continue

    # Вычисляем evacuated_ratio вручную
    base_vals = df_base["evacuated"].iloc[-N_LAST:] / TOTAL_AGENTS * 100
    fix_vals = df_fix["evacuated"].iloc[-N_LAST:] / TOTAL_AGENTS * 100

    stat, p = mannwhitneyu(base_vals, fix_vals, alternative="two-sided")
    delta = fix_vals.mean() - base_vals.mean()

    print(f"{algo}: p = {p:.5f} ✅ | Δmean = {delta:.2f} %")
