import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === ПАРАМЕТРЫ ===
METRICS = ['evacuated_ratio', 'avg_steps', 'duration', 'evac_min_ratio', 'evac_max_ratio']
FOLDERS = {
    'simple': Path('метрики 1000\дефолт'),
    'fire spawn fix': Path('метрики 1000\фикс спавна огня'),
}
ALGORITHMS = [
    "DuelingDDQN",
    "DuelingDDQNPrioritized",
    "NoisyDuelingDDQN",
    "QRDQN"
]
TOTAL_AGENTS = 1200  # Замените на реальное количество агентов
SMOOTH_WINDOW = 50

# === ЗАГРУЗКА ДАННЫХ ===
def load_data(folder: Path, algo: str):
    df = pd.read_csv(folder / f"metrics_{algo}.csv")
    df['evacuated_ratio'] = df['evacuated'] / TOTAL_AGENTS
    df['evac_min_ratio'] = df['evac_min'] / 48
    df['evac_max_ratio'] = df['evac_max'] / 48
    
    return df

# === ПОСТРОЕНИЕ ГРАФИКОВ ===
for metric in METRICS:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Метрика: {metric}', fontsize=16)

    for i, algo in enumerate(ALGORITHMS):
        row, col = divmod(i, 2)
        ax = axs[row][col]

        for label, folder in FOLDERS.items():
            df = load_data(folder, algo)
            smooth = df[metric].rolling(window=SMOOTH_WINDOW).mean()
            ax.plot(df['episode'], smooth, label=label)

        ax.set_title(algo)
        ax.set_xlabel('Эпизод')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'comparison_{metric}.png')
    plt.show()
