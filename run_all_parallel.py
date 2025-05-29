# run_all_parallel.py
import subprocess
import json
from pathlib import Path

ALGORITHMS = [
    "DuelingDDQN",
    "QRDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN"
]

CONFIG_PATH = "config.json"

# Загружаем и обновляем конфиг 
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Перезаписываем общий конфиг
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

# Запускаем каждый алгоритм параллельно
processes = []
for algo in ALGORITHMS:
    cmd = ["python", "main.py", "--config", CONFIG_PATH]
    env_cfg = cfg.copy()
    env_cfg["ALGORITHM"] = algo
    env_cfg["METRICS_CSV"] = f"metrics_{algo}.csv"
    env_cfg["VIDEO_PATH"] = f"video_{algo}.mp4"

    # создаём временный конфиг для каждого процесса
    tmp_cfg_path = f"config_tmp_{algo}.json"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        json.dump(env_cfg, f, indent=2)

    log_file = open(f"log_{algo}.txt", "w")
    p = subprocess.Popen(["python", "main.py", "--config", tmp_cfg_path], stdout=log_file, stderr=log_file)
    processes.append(p)

# Ждём завершения всех процессов
for p in processes:
    p.wait()

print("✅ Все 4 алгоритма завершили обучение.")
