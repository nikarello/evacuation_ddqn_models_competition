# core/replay_buffer_batched.py
import torch

class ReplayBufferBatched:
    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device   = device
        self.pos, self.size = 0, 0
        self.S = self.A = self.R = self.NS = self.D = None

    def __len__(self): return self.size

    def push(self, S, A, R, NS, D):
        B = S.size(0)
        if self.S is None:
            shape = (self.capacity,) + S.shape[1:]
            self.S  = torch.zeros(shape, dtype=S.dtype , device=self.device)
            self.NS = torch.zeros_like(self.S)
            self.A  = torch.zeros(self.capacity, 1, dtype=A.dtype , device=self.device)
            self.R  = torch.zeros(self.capacity,     dtype=R.dtype , device=self.device)
            self.D  = torch.zeros(self.capacity,     dtype=D.dtype , device=self.device)

        idxs = (torch.arange(B, device=self.device) + self.pos) % self.capacity
        self.S [idxs] = S
        self.A [idxs] = A
        self.R [idxs] = R
        self.NS[idxs] = NS
        self.D [idxs] = D

        self.pos  = (self.pos + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.S [idxs],
            self.A [idxs],
            self.R [idxs],
            self.NS[idxs],
            self.D [idxs],
        )
