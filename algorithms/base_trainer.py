# algorithms/base_trainer.py
import abc, time, torch

import csv, pathlib, pandas as pd, matplotlib.pyplot as plt
import math, numpy as np

import imageio, os
from PIL import Image
import numpy as np
from datetime import datetime

from environment import stack_envs
from core import batched_update_fire_exit, batched_update_agents, \
                batched_get_views, batched_step

class BaseTrainer(abc.ABC):
    def __init__(self, envs, cfg):
        self.envs=envs; self.cfg=cfg
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = None
        self.target = None
        # 🔧 Универсальные параметры
        self.in_ch      = 8 # количество каналов обзора
        self.view       = cfg["VIEW_SIZE"]
        self.gamma      = cfg["GAMMA"]
        self.batch_size = cfg["BATCH_SIZE"]
        self.memory_size= cfg.get("MEMORY_SIZE", 100_000)
        self.eps_start  = cfg.get("EPSILON_START", 1.0)
        self.eps_min    = cfg.get("EPSILON_MIN", 0.1)
        self.eps_decay  = cfg.get("EPSILON_DECAY", 0.995)
        self.n_envs     = cfg["NUM_ENVS"]
        self.max_steps  = cfg["MAX_STEPS_PER_EPISODE"]
        self.target_update_freq = cfg.get("TARGET_UPDATE_FREQ", 5)
        self.grid_size  = cfg["GRID_SIZE"]
        self.agent_specs= cfg["AGENT_SPECS"]

        self.lr         = cfg["LEARNING_RATE"]
        # 🔁 Learning Rate decay
        self.lr_min     = cfg.get("LEARNING_RATE_MIN", 1e-5)
        self.lr_decay   = (self.lr_min / self.lr) ** (1.0 / cfg["NUM_EPISODES"])

    # ---- абстрактные ----
    @abc.abstractmethod
    def build_model(self): ...
    @abc.abstractmethod
    def select_actions(self, views, mask): ...
    @abc.abstractmethod
    def learn_step(self): ...
    @abc.abstractmethod
    def after_episode(self, ep): ...
    # ---- общий цикл ----

    
    def train(self):

        B = len(self.envs)
        for e in self.envs:
            e.reset()

        self._init_metrics()
        

        for ep in range(self.cfg["NUM_EPISODES"]):
            start = time.time()
            pos, alive, know, hp, sz, sp, fire, exits, leader_pos, leader_alive = stack_envs(self.envs)

            pos        = pos.to(self.device)
            alive      = alive.to(self.device)
            know       = know.to(self.device)
            hp         = hp.to(self.device)
            sz         = sz.to(self.device)
            sp         = sp.to(self.device)
            fire       = fire.to(self.device)
            exits      = exits.to(self.device)
            leader_pos = leader_pos.to(self.device)
            leader_alive = leader_alive.to(self.device)

            done = torch.zeros(B, dtype=torch.bool, device=self.device)

            exited_mask = torch.zeros((B, self.num_agents), dtype=torch.bool, device=self.device)
            died_mask   = torch.zeros_like(exited_mask)
            evacuated_mask = torch.zeros_like(alive)
            exit_step   = torch.full_like(exited_mask, -1, dtype=torch.int32)
            hp_at_exit  = torch.zeros_like(exited_mask, dtype=torch.float)

            total_reward = 0.0

            for step in range(1, self.max_steps + 1):
                # 🔥 Обновляем огонь и выход
                fire, exits = batched_update_fire_exit(fire, exits)

                # Исправленный вызов (цикл)
                for env in self.envs:
                    env.step_leaders()

                # 🧱 Обновляем карты агентов (только живые)
                ag, szm, spm, inf, leader_map = batched_update_agents(
                    pos, sz, sp, know, alive, self.grid_size, 
                    leader_pos, leader_alive, self.envs[0].leader_size
                )

                # 👁️ Получаем обзоры
                views = batched_get_views(
                    leader_map, ag, fire, exits, szm, spm, inf, pos, self.view
                )

                # 🧠 Маска агентов, готовых к действию
                mask = alive & (~know)

                # DEBUG: показываем, сколько агентов реально участвует в обучении
                #num_active = mask.sum().item()
                #print(f"[step {step:>3}] active agents: {num_active} | views shape: {views.shape}")

                # 🎮 Выбор действия
                actions = self.select_actions(views, mask)

                # 🚶‍♂️ Шаг среды
                next_pos, rewards, dones, alive, hp, fire, exits, died, exs = \
                    batched_step(pos, actions, sz, sp, fire, exits, hp, leader_pos, leader_alive, self.envs[0].leader_size)

                # 🚪 Зафиксировать ушедших
                evacuated_mask |= exs
                alive &= ~evacuated_mask

                # 📊 Получить следующее состояние
                next_ag, next_szm, next_spm, next_inf, next_leader_map = batched_update_agents(
                    next_pos, sz, sp, know, alive, self.grid_size,
                    leader_pos, leader_alive, self.envs[0].leader_size
                )


                next_views = batched_get_views(
                    next_leader_map, next_ag, fire, exits, next_szm, next_spm, next_inf, next_pos, self.view
                )



                # 💾 Сохранить переходы в буфер
                for b in range(B):
                    idxs = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
                    if idxs.numel() == 0:
                        continue

                    s_batch  = views[b, idxs]
                    a_batch  = actions[b, idxs].unsqueeze(1)
                    r_batch  = rewards[b, idxs]
                    ns_batch = next_views[b, idxs]
                    d_batch = dones[b].expand(idxs.shape[0]).float()



                    self.store_transition(s_batch, a_batch, r_batch, ns_batch, d_batch)



                # метрики
                total_reward += rewards.sum().item()
                newly_exit = exs & (~exited_mask)
                newly_die  = died  & (~died_mask)
                exited_mask |= newly_exit
                died_mask   |= newly_die
                exit_step[newly_exit]  = step
                hp_at_exit[newly_exit] = hp[newly_exit]

                pos = next_pos
                done |= dones

                l = self.learn_step()
                if l is not None and l > 0:
                    self.loss_sum += l
                    self.loss_cnt += 1

                if done.all():
                    break
                
            total_agents = self.num_agents * self.n_envs  # N × ENV
            evac_cnt = int(exited_mask.sum().item())
            died_cnt = total_agents - evac_cnt
            avg_st   = float(exit_step[exited_mask].float().mean().item()) if evac_cnt else math.nan
            avg_hp   = float(hp_at_exit[exited_mask].mean().item())         if evac_cnt else math.nan
            avg_loss = self.loss_sum / self.loss_cnt if self.loss_cnt else math.nan
            evac_counts = exited_mask.sum(dim=1)  # по всем ENV
            min_evac = int(evac_counts.min().item())
            max_evac = int(evac_counts.max().item())
            try:
                curr_lr = self.opt.param_groups[0]["lr"]
            except Exception:
                curr_lr = float("nan")


            self.metrics.append({
                "episode":   ep + 1,
                "reward":    float(total_reward),
                "evacuated": evac_cnt,
                "died":      died_cnt,
                "avg_steps": avg_st,
                "avg_hp":    avg_hp,
                "epsilon":   float(getattr(self, "epsilon", float("nan"))),
                "loss":      avg_loss,
                "duration":  time.time() - start,
                "lr": curr_lr,
                "evac_min": min_evac,
                "evac_max": max_evac,
            })

            self.loss_sum, self.loss_cnt = 0.0, 0
            self.after_episode(ep)
            
            # 🔽 Automatic learning rate decay
            if hasattr(self, "opt") and self.opt is not None:
                for g in self.opt.param_groups:
                    g["lr"] = max(self.lr_min, g["lr"] * self.lr_decay)

            curr_lr = self.opt.param_groups[0]["lr"] if hasattr(self, "opt") else float("nan")
            if ep == 0:
                print("ALGORITHM:", self.cfg.get("ALGORITHM", "UnknownAlgorithm"))

            print(f"EP {ep + 1}/{self.cfg['NUM_EPISODES']} | time {time.time() - start:>5.2f}s | "
                f"reward {total_reward:>8.1f} | evac {evac_cnt:>3} | "
                f"died {died_cnt:>3} | eps {getattr(self, 'epsilon', float('nan')):>4.3f} | "
                f"lr {curr_lr:.5f} | loss {avg_loss:>6.3f}", flush=True)

        self._save_metrics_csv()
        
        self._plot_metrics()

        for i in range(4):
            if i >= len(self.envs): break 
            self._render_episode(env_idx=i)
            
    def _init_metrics(self):
        self.metrics = []                       
        self.loss_sum, self.loss_cnt = 0.0, 0
        # для эвакуации / смертей
        env0 = self.envs[0]
        self.num_agents = env0.N              

    def _save_metrics_csv(self):
        fname = self.cfg.get("METRICS_CSV", "metrics.csv")
        df = pd.DataFrame(self.metrics)
        out = pathlib.Path(fname)
        df.to_csv(out, index=False)
        print("[done] metrics saved ->", str(out.resolve().absolute()))


    def _plot_metrics(self):
        df = pd.DataFrame(self.metrics)
        plt.figure(figsize=(12, 12))

        algo = self.cfg.get("ALGORITHM", "Algorithm")
        plt.suptitle(f"{algo} — Training Metrics", fontsize=16)

        plt.subplot(4, 2, 1); plt.plot(df["episode"], df["reward"]); plt.title("Reward")

        plt.subplot(4, 2, 2); plt.plot(df["episode"], df["loss"]);    plt.title("Loss")

        plt.subplot(4, 2, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Левая ось: абсолютные значения
        ax1.plot(df["episode"], df["evacuated"], label="Evacuated", color="tab:green")
        ax1.plot(df["episode"], df["died"], label="Died", color="tab:red")
        ax1.set_ylabel("Count")
        ax1.legend(loc="upper left")

        # Правая ось: процент эвакуированных
        total_agents = self.num_agents * self.n_envs
        evac_percent = df["evacuated"] / total_agents * 100
        ax2.plot(df["episode"], evac_percent, label="% Evacuated", color="tab:blue", linestyle="--")
        ax2.set_ylabel("% Evacuated")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right")

        plt.title("Evacuated / Died / % Evacuated")

        plt.subplot(4, 2, 4)
        plt.plot(df["episode"], df["evac_min"], label="min evac")
        plt.plot(df["episode"], df["evac_max"], label="max evac")
        plt.title("Min/Max Evacuated per Env")
        plt.legend()

        plt.subplot(4, 2, 5); plt.plot(df["episode"], df["avg_steps"]); plt.title("Avg steps to exit")
        plt.subplot(4, 2, 6); plt.plot(df["episode"], df["avg_hp"]); plt.title("Avg HP at exit")
        plt.subplot(4, 2, 7); plt.plot(df["episode"], df["epsilon"]); plt.title("Epsilon")
        plt.subplot(4, 2, 8); plt.plot(df["episode"], df["lr"]); plt.title("lr (Learning Rate)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    # -----------------------------------------------------------------------
    # UNIVERSAL RENDER ------------------------------------------------------
    # -----------------------------------------------------------------------
    def _render_episode(
        self,
        env_idx: int = 0,
        max_steps: int | None = None,
        out_dir: str = "videos",
        scale: int = 10,
    ):
        """Проиграть один эпизод без ε-шума и записать MP4-ролик.

        • env_idx   – какую среду взять (0-based)  
        • max_steps – обрезка по шагам (None → self.max_steps)  
        • scale     – во сколько раз «увеличить пиксели» для красивой картинки
        """
        os.makedirs(out_dir, exist_ok=True)
        env       = self.envs[env_idx]
        algo_name = self.__class__.__name__.replace("Trainer", "")
        fname     = os.path.join(
            out_dir,
            f"{algo_name}_env{env_idx+1:02d}.mp4"
        )

        # ---------- подготовка сети к дет-режиму -------------
        prev_eps          = getattr(self, "epsilon", None)
        self.epsilon      = 0.0                # полностью greedy
        noisy             = hasattr(self.online, "reset_noise")
        self.online.eval()
        self.target.eval()
        if noisy:                                 # фиксируем шум один раз
            self.online.reset_noise()
            self.target.reset_noise()

        env.reset()
        grid = self.grid_size
        frames = []

        max_steps = max_steps or self.max_steps
        pos, alive, know, hp, sz, sp, fire, exit_m, leader_pos, leader_alive = stack_envs([env])


        for _ in range(max_steps):

            env.step_leaders()

            ag, szm, spm, inf, leader_map = batched_update_agents(
                pos, sz, sp, know, alive, self.grid_size,
                env.leader_positions.unsqueeze(0),
                env.leader_alive.unsqueeze(0),
                env.leader_size
            )
            views = batched_get_views(
                leader_map, ag, fire, exit_m, szm, spm, inf, pos, self.view
            )

            actions = self.select_actions(views, alive & (~know))


            pos, _, done, alive, hp, fire, exit_m, _, _ = batched_step(
                pos, actions, sz, sp, fire, exit_m, hp,
                env.leader_positions.unsqueeze(0),  # [1, L, 2]
                env.leader_alive.unsqueeze(0),      # [1, L]
                env.leader_size
            )

            

            # Обновляем карту агентов уже с обновлённым alive
            ag, szm, spm, inf, leader_map = batched_update_agents(
                pos, sz, sp, know, alive, self.grid_size, 
                env.leader_positions.unsqueeze(0),  # [1, num_leaders, 2]
                env.leader_alive.unsqueeze(0),      # [num_leaders]
                env.leader_size
            )

            # --- рисуем -----------------------------------------------------------------
            rgb = np.zeros((grid, grid, 3), dtype=np.uint8)
            rgb[exit_m[0].cpu().numpy() == 1] = (0, 255, 0)          # выход – зелёный
            rgb[fire  [0].cpu().numpy() == 1] = (255, 0, 0)          # огонь – красный
            ys, xs = np.where(ag[0].cpu().numpy() > 0.5)             # агенты – синий
            rgb[ys, xs] = (0, 0, 255)

            ys, xs = np.where(leader_map[0].cpu().numpy() > 0.5)
            rgb[ys, xs] = (255, 255, 0)  # лидеры — жёлтые

            pil = Image.fromarray(rgb).resize(
                (grid * scale, grid * scale), Image.NEAREST
            )
            frames.append(np.asarray(pil))

            if done.item():
                break

            # распространение огня – ОБЯЗАТЕЛЬНО в конце цикла,
            # иначе последний кадр не совпадёт с «шагом среды»
            fire, exit_m = batched_update_fire_exit(fire, exit_m)

        # ---------- запись video ----------------------------------------------------------
        with imageio.get_writer(
            fname, fps=5, codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline"]
        ) as vid:
            for f in frames:
                vid.append_data(f)

        # ---------- восстановление состояний ---------------------------------------------
        if prev_eps is not None:
            self.epsilon = prev_eps
        print(f"[done] video saved -> {fname!s}")
