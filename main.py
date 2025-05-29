import json, argparse
from environment import Environment
import torch, gc

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Алгоритмы с унифицированными ключами (lowercase, без подчёркиваний)
ALGOS = {
    "duelingddqn":            "algorithms.dueling_ddqn.DuelingDDQNTrainer",
    "qrdqn":                  "algorithms.qr_dqn.QRDQNTrainer",
    "duelingddqnprioritized":"algorithms.dueling_ddqn_prioritized.DuelingDDQNPrioritizedTrainer",
    "noisyduelingddqn":       "algorithms.noisy_dueling_ddqn.NoisyDuelingDDQNTrainer"
}

def load_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # === Автоматический расчёт EPSILON_DECAY ===
    N = cfg["NUM_EPISODES"] * (cfg["MAX_STEPS_PER_EPISODE"] / 3)
    cfg["EPSILON_DECAY"] = (cfg["EPSILON_MIN"] / cfg["EPSILON_START"]) ** (1 / N)

    return cfg

def create_trainer(name, envs, cfg):
    key = name.lower().replace("_", "")  # унификация: lowercase + без подчёркиваний
    if key not in ALGOS:
        raise ValueError(f"Unknown algorithm: {name}")
    module_path, cls_name = ALGOS[key].rsplit(".", 1)
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)(envs, cfg)

def default_exit(grid):
    width     = max(1, int(grid * 0.25))
    thickness = 4
    xs = range(grid - width, grid)
    ys = range(grid - thickness, grid)
    return [(x, y) for y in ys for x in xs]

def main():
    cfg = load_cfg()

    exit_pos = cfg.get("EXIT_POS") or default_exit(cfg["GRID_SIZE"])

    envs = [Environment(cfg["GRID_SIZE"],
                        cfg["AGENT_SPECS"],
                        num_fires=1,
                        exit_pos=exit_pos, num_leaders=cfg.get("NUM_LEADERS", 0))
            for _ in range(cfg["NUM_ENVS"])]

    trainer = create_trainer(cfg["ALGORITHM"], envs, cfg)
    trainer.train()


    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
