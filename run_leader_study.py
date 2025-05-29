# run_leader_study.py

import subprocess
import json
from pathlib import Path
import time

ALGORITHMS = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN"
]

LEADER_COUNTS = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 20]
CONFIG_PATH = "config.json"
SAVE_DIR = Path("liders_check")
SAVE_DIR.mkdir(exist_ok=True)

LOGS_DIR = SAVE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

CONFIGS_DIR = SAVE_DIR / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

# Загружаем базовый конфиг
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    base_cfg = json.load(f)

# Сохраняем контрольные параметры
NUM_EPISODES = 50
MAX_STEPS    = base_cfg["MAX_STEPS_PER_EPISODE"]

# По очереди перебираем количество лидеров
for n_leaders in LEADER_COUNTS:
    print(f"\n=== Запуск для NUM_LEADERS = {n_leaders} ===")
    processes = []

    for algo in ALGORITHMS:
        cfg = base_cfg.copy()
        cfg["ALGORITHM"] = algo
        cfg["NUM_LEADERS"] = n_leaders

        # 🧷 Явно задаём критичные параметры
        cfg["NUM_EPISODES"] = NUM_EPISODES
        cfg["MAX_STEPS_PER_EPISODE"] = MAX_STEPS

        cfg["METRICS_CSV"] = str(SAVE_DIR / f"leaders_{n_leaders}_{algo}.csv")
        cfg["VIDEO_PATH"] = str(SAVE_DIR / f"leaders_{n_leaders}_{algo}.mp4")

        tmp_cfg_path = CONFIGS_DIR / f"tmp_config_{algo}_{n_leaders}.json"
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        log_path = LOGS_DIR / f"log_leaders_{n_leaders}_{algo}.txt"
        log_file = open(log_path, "w")

        p = subprocess.Popen(["python", "main.py", "--config", str(tmp_cfg_path)],
                             stdout=log_file, stderr=log_file)
        processes.append(p)

    for p in processes:
        p.wait()

    print(f" Все 4 алгоритма завершили обучение для NUM_LEADERS = {n_leaders}\n")

print(" Все запуски завершены.")
