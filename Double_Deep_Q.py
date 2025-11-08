import math
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------- Utils ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ---------- Replay Buffer ----------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        states = np.vstack([b.state for b in batch]).astype(np.float32)
        next_states = np.vstack([b.next_state for b in batch]).astype(np.float32)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)


# ---------- Dueling Q-Network ----------
class DuelingMLP(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.SiLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden[1], hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], n_actions)  # linear head (no softmax!)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden[1], hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], 1)          # state-value
        )

    def forward(self, x):  # x: [B, state_dim]
        z = self.feature(x)
        adv = self.advantage(z)              # [B, A]
        val = self.value(z)                  # [B, 1]
        # Dueling combine: Q = V + (A - mean(A))
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


# ---------- Epsilon Scheduler ----------
class EpsilonScheduler:
    def __init__(self, eps_start=1.0, eps_end=0.10, eps_decay_steps=500_000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.step = 0

    def value(self):
        # exponential schedule is common, here linear is fine:
        frac = min(1.0, self.step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def update(self):
        self.step += 1


# ---------- DDQN Agent ----------
class DDQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma=0.99,
        lr=3e-4,
        batch_size=256,
        tau=0.005,                # soft target update
        eps_start=1.0,
        eps_end=0.10,
        eps_decay_steps=500_000,
        buffer_capacity=100_000,
        grad_clip_norm=10.0,
        seed=42,
        device=None,
    ):
        set_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm

        self.q_net = DuelingMLP(state_dim, n_actions).to(self.device)
        self.target_net = DuelingMLP(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, n_actions)
        self.eps = EpsilonScheduler(eps_start, eps_end, eps_decay_steps)

        self.learn_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        # epsilon-greedy
        eps = self.eps.value()
        self.eps.update()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = to_tensor(state, self.device).unsqueeze(0)  # [1, state_dim]
        q = self.q_net(s)                               # [1, n_actions]
        return int(torch.argmax(q, dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def soft_update(self):
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = to_tensor(batch.state, self.device)              # [B, state_dim]
        next_states = to_tensor(batch.next_state, self.device)    # [B, state_dim]
        actions = torch.as_tensor(batch.action, device=self.device, dtype=torch.long)  # [B]
        rewards = to_tensor(batch.reward, self.device)            # [B]
        dones = to_tensor(batch.done, self.device)                # [B], 1.0 if done else 0.0

        # Current Q(s,a)
        q_values = self.q_net(states)                             # [B, A]
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            # Double DQN:
            # next action from online net, valued by target net
            q_next_eval = self.q_net(next_states)                 # [B, A]
            next_actions = torch.argmax(q_next_eval, dim=1)       # [B]

            q_next_target = self.target_net(next_states)          # [B, A]
            q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # [B]

            target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = self.criterion(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optim.step()

        self.soft_update()
        self.learn_steps += 1

        with torch.no_grad():
            td_error = (q_sa - target).abs().mean().item()
            q_mean = q_values.mean().item()

        return {"loss": float(loss.item()), "td_error": td_error, "q_mean": q_mean}

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict()
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.target_net.eval()
