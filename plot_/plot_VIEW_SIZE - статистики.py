import pandas as pd
import scipy.stats as stats
from pathlib import Path
import re

# === Настройки ===
FOLDER = Path("smoke_check")
TOTAL_AGENTS = 480
csv_files = list(FOLDER.glob("smoke_*.csv"))

records = []

# === Загрузка и предобработка данных ===
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

if not records:
    print("❌ Нет подходящих CSV-файлов.")
    exit()

# === Объединение ===
df_all = pd.concat(records, ignore_index=True)

# === Kruskal–Wallis test ===
print("=== 🧪 Kruskal–Wallis H-тест по evacuated_pct ===\n")

algorithms = df_all["algorithm"].unique()

for algo in sorted(algorithms):
    df_algo = df_all[df_all["algorithm"] == algo]
    groups = []
    labels = []

    for view_size in sorted(df_algo["view_size"].unique()):
        vals = df_algo[df_algo["view_size"] == view_size]["evacuated_pct"].values
        if len(vals) >= 10:  # хотя бы 10 эпизодов для устойчивости
            groups.append(vals)
            labels.append(f"VIEW_SIZE = {view_size}")

    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        result = "✅ значимо" if p < 0.05 else "— незначимо"
        print(f"{algo}: H = {stat:.4f}, p = {p:.5f} {result}")
    else:
        print(f"{algo}: недостаточно данных для анализа")
