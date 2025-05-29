# algorithms/dueling_ddqn_prioritized.py
import math, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from algorithms.base_trainer import BaseTrainer
from core import device
from torch.cuda.amp import autocast, GradScaler

# ======= GPU Prioritized Replay Buffer (batched) =======
class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, prior_eps=1e-5):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.prior_eps = prior_eps
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        # Хранилища
        self.S = None
        self.A = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.R = torch.zeros(capacity, dtype=torch.float, device=device)
        self.NS = None
        self.D = torch.zeros(capacity, dtype=torch.float, device=device)
        self.priorities = torch.zeros(capacity, dtype=torch.float, device=device)

    def __len__(self): return self.size

    def push(self, S, A, R, NS, D):
        """Добавление батчем"""
        B = S.shape[0]

        if self.S is None:
            self.S = torch.zeros((self.capacity,) + S.shape[1:], dtype=torch.float, device=self.device)
            self.NS = torch.zeros_like(self.S)

        idxs = (torch.arange(B) + self.pos) % self.capacity
        self.S[idxs] = S.to(self.device)
        self.NS[idxs] = NS.to(self.device)
        self.A[idxs] = A
        self.R[idxs] = R
        self.D[idxs] = D
        self.priorities[idxs] = self.max_priority

        self.pos = (self.pos + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size, beta=0.4):
        probs = self.priorities[:self.size].pow(self.alpha)
        probs /= probs.sum()

        idxs = torch.multinomial(probs, batch_size, replacement=False)

        weights = (self.size * probs[idxs]).pow(-beta)
        weights /= weights.max()

        return (
            self.S[idxs],
            self.A[idxs],
            self.R[idxs],
            self.NS[idxs],
            self.D[idxs],
            idxs,
            weights
        )

    def update_priorities(self, idxs, td_errors):
        td = torch.abs(torch.tensor(td_errors, dtype=torch.float, device=self.device))
        age_bonus = 1e-3 * (self.pos / (self.capacity + 1))
        new_prior = td + self.prior_eps + age_bonus
        self.priorities[idxs] = new_prior
        self.max_priority = max(self.max_priority, new_prior.max().item())


# ======= Dueling CNN =======
class DuelingCNN(nn.Module):
    def __init__(self, in_ch:int, n_act:int, view:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.GroupNorm(8, 64),
            nn.Conv2d(64,128,3, 1, 1), nn.ReLU(), nn.GroupNorm(8, 128),
            nn.Flatten())
        feat = 128 * view * view
        self.val = nn.Sequential(nn.Linear(feat,256), nn.ReLU(), nn.Linear(256,1))
        self.adv = nn.Sequential(nn.Linear(feat,256), nn.ReLU(), nn.Linear(256, n_act))

    def forward(self, x):
        f = self.conv(x)
        v = self.val(f)
        a = self.adv(f)
        return v + a - a.mean(1, keepdim=True)


# ======= Dueling DDQN + PER Trainer =======
class DuelingDDQNPrioritizedTrainer(BaseTrainer):
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)

        self.online = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        self.target = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.replay = PrioritizedReplayBuffer(
            self.memory_size, self.device, alpha=cfg.get("PER_ALPHA", 0.6)
        )
        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)

        self.beta_start  = cfg.get("PER_BETA_START", 0.4)
        self.beta_frames = cfg.get("PER_BETA_FRAMES", cfg["NUM_EPISODES"] * self.max_steps)
        self.frame_idx   = 0
        self.beta        = self.beta_start

        self.epsilon = self.eps_start

        self.scaler = GradScaler()

    def select_actions(self, views, mask):
        B, N, C, V, _ = views.shape
        flat = views.reshape(B * N, C, V, V)
        with torch.no_grad():
            q = self.online(flat)
        greedy = q.argmax(1).view(B, N)

        rand = torch.randint(0, 4, (B, N), device=self.device)
        eps_mask = (torch.rand(B, N, device=self.device) < self.epsilon) & mask
        return torch.where(eps_mask, rand, greedy)

    def learn_step(self):
        if len(self.replay) < self.batch_size:
            return None

        self.frame_idx += 1
        self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        S, A, R, NS, D, idxs, w = self.replay.sample(self.batch_size, beta=self.beta)

        with autocast():
            q_curr = self.online(S).gather(1, A).squeeze(1)
            with torch.no_grad():
                best = self.online(NS).argmax(1, keepdim=True)
                q_next = self.target(NS).gather(1, best).squeeze(1)
                tgt = R + self.gamma * q_next * (1.0 - D)

            td_errors = tgt - q_curr
            loss = (w * td_errors.pow(2)).mean()

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        self.replay.update_priorities(idxs, td_errors.detach().cpu().numpy())

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return loss.item()

    def after_episode(self, ep):
        if (ep + 1) % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

    def store_transition(self, s, a, r, ns, d):
        """Теперь принимает батчи"""
        self.replay.push(s, a, r, ns, d)

    def build_model(self):
        return self.online, self.target
