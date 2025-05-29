import subprocess, json, threading, time
from pathlib import Path
from tqdm import tqdm
from itertools import islice

ALGORITHMS = [
    "QRDQN",
    "DuelingDDQNPrioritized", "NoisyDuelingDDQN", "DuelingDDQN"
]

SMOKE_LEVELS = [11, 13, 15, 17, 19, 21]
CONFIG_PATH = "config.json"
SAVE_DIR = Path("smoke_check")
SAVE_DIR.mkdir(exist_ok=True)
(SAVE_DIR / "logs").mkdir(exist_ok=True)
(SAVE_DIR / "configs").mkdir(exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    base_cfg = json.load(f)

NUM_EPISODES = 100

MAX_STEPS = base_cfg["MAX_STEPS_PER_EPISODE"]
total_runs = len(SMOKE_LEVELS) * len(ALGORITHMS)
progress_bar = tqdm(total=total_runs, desc="🔥 Smoke study progress", ncols=100)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def stream_to_file(pipe, file):
    for line in iter(pipe.readline, b''):
        file.write(line.decode('utf-8', errors='ignore'))
        file.flush()

for view_size in SMOKE_LEVELS:
    print(f"\n=== Запуск для VIEW_SIZE = {view_size} ===")

    for algo_batch in batched(ALGORITHMS, 1):  # 🔁 по 2 алгоритма
        processes = []

        for algo in algo_batch:
            cfg = base_cfg.copy()
            cfg["ALGORITHM"] = algo
            cfg["VIEW_SIZE"] = view_size
            cfg["NUM_EPISODES"] = NUM_EPISODES
            cfg["MAX_STEPS_PER_EPISODE"] = MAX_STEPS
            cfg["METRICS_CSV"] = str(SAVE_DIR / f"smoke_{view_size}_{algo}.csv")
            cfg["VIDEO_PATH"] = str(SAVE_DIR / f"smoke_{view_size}_{algo}.mp4")

            tmp_cfg_path = SAVE_DIR / "configs" / f"tmp_smoke_{algo}_{view_size}.json"
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            log_path = SAVE_DIR / "logs" / f"log_smoke_{view_size}_{algo}.txt"
            log_file = open(log_path, "w", encoding='utf-8')

            print(f"🚀 Запуск {algo} (VIEW_SIZE={view_size})...")
            try:
                p = subprocess.Popen(["python", "main.py", "--config", str(tmp_cfg_path)],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                t = threading.Thread(target=stream_to_file, args=(p.stdout, log_file), daemon=True)
                t.start()
                processes.append((p, log_file, t))
            except Exception as e:
                print(f"❌ Ошибка при запуске {algo} VIEW={view_size}: {e}")
                progress_bar.update(1)

        # ⏱️ Ждём завершения только текущей пары
        while processes:
            time.sleep(5)
            still_running = []
            for p, log_file, thread in processes:
                if p.poll() is None:
                    still_running.append((p, log_file, thread))
                else:
                    thread.join()
                    log_file.close()
                    progress_bar.update(1)
            processes = still_running

progress_bar.close()
print("\n✅ Все запуски по гипотезе задымления завершены.")
