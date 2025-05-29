# algorithms/dueling_ddqn.py
import torch, torch.nn as nn, torch.optim as optim
from algorithms.base_trainer import BaseTrainer
from core_modules.replay_buffer_batched import ReplayBufferBatched as ReplayBuffer
from core import batched_update_fire_exit, batched_update_agents, \
                 batched_get_views, batched_step     # последнюю добавим чуть позже
from torch.cuda.amp import autocast, GradScaler
# ===== модель =====
class DuelingCNN(nn.Module):
    def __init__(self,in_ch, actions, view):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,1,1), nn.ReLU(), nn.GroupNorm(8, 32),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.GroupNorm(8, 64),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.GroupNorm(8, 128),
            nn.Flatten()
        )

        dim=128*view*view
        self.v=nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Linear(256,1))
        self.a=nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Linear(256,actions))
    def forward(self,x):
        f=self.conv(x)
        v=self.v(f); a=self.a(f)
        return v + a - a.mean(1,keepdim=True)

# ===== тренер =====
class DuelingDDQNTrainer(BaseTrainer):
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)
        self.online, self.target = self.build_model()
        
        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        self.buf = ReplayBuffer(self.memory_size, self.device)

        self.epsilon = self.eps_start
        self.scaler = GradScaler()

    def build_model(self):
        m = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        t = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        t.load_state_dict(m.state_dict())
        t.eval()
        return m, t
    
    def store_transition(self, s, a, r, ns, d):
        """
        Кладём батч переходов в обычный ReplayBufferBatched.
        • s, ns :  [B, C, V, V]
        • a     :  [B, 1]
        • r, d  :  [B]
        """
        # Буфер уже хранится на self.device, поэтому .to(...) обычно не нужны,
        # но оставим на случай, если тензоры лежат на CPU.
        self.buf.push(
            s.to(self.device, non_blocking=True),
            a.to(self.device, non_blocking=True),
            r.to(self.device, non_blocking=True),
            ns.to(self.device, non_blocking=True),
            d.to(self.device, non_blocking=True),
        )
        
    def select_actions(self, views, mask):
        B, N, *rest = views.shape
        flat = views.reshape(B * N, *rest)

        with torch.no_grad():
            q = self.online(flat)
        q = q.view(B, N, -1).argmax(dim=2)

        rand = torch.randint(0, 4, (B, N), device=self.device)
        choose = torch.where(torch.rand_like(q.float()) < self.epsilon, rand, q)

        acts = torch.zeros_like(q)
        acts[mask] = choose[mask]
        return acts

    def learn_step(self):
        if len(self.buf) < self.batch_size:
            return None

        S, A, R, NS, D = self.buf.sample(self.batch_size)

        if torch.isnan(S).any() or torch.isnan(R).any() or torch.isnan(NS).any():
            print("⚠️ Found NaNs in input batch!")

        with autocast():  # 🔧 Включаем AMP
            qc = self.online(S).gather(1, A).squeeze()
            with torch.no_grad():
                best = self.online(NS).argmax(1, keepdim=True)
                qn = self.target(NS).gather(1, best).squeeze()
            tgt = R + self.gamma * qn * (1 - D)

            if torch.isnan(tgt).any():
                print("❌ tgt has NaNs")
            if torch.isnan(qc).any():
                print("❌ qc has NaNs")

            loss = nn.functional.mse_loss(qc, tgt)

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        if hasattr(self, "epsilon"):
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        return loss.item()


    def after_episode(self, ep):
        if (ep + 1) % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
