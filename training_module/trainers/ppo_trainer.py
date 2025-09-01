# training_module/policies/ppo_agent.py
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Torch opcional (para CPU sirve igualmente). Si no hay torch, se usa un stub.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class PPOConfig:
    symbol: str = "BTCUSDT"
    obs_features: int = 64 * 32   # (W*F) si aplanas obs [W,F] → [W*F]
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    device: str = "cpu"


# ============== Implementación Torch ==============
class _ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_actions: int = 3):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


class PPOAgentTorch:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = _ActorCritic(cfg.obs_features, cfg.hidden).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def act(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        # obs_np: [W,F] → aplanado
        x = torch.from_numpy(obs_np.reshape(-1)).float().unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def evaluate(self, obs_batch: torch.Tensor, act_batch: torch.Tensor):
        logits, values = self.model(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(act_batch)
        entropy = dist.entropy()
        return logprobs, values, entropy

    def update(self, batch: Dict[str, torch.Tensor]):
        """
        batch keys:
          obs: [B, W*F], actions: [B], logprobs: [B], returns: [B], advantages: [B]
        """
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        returns = batch["returns"]
        adv = batch["advantages"]

        logprobs, values, entropy = self.evaluate(obs, actions)
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "loss": float(loss.item()),
        }

    # --- Guardado/Carga en PPO_models/PPO_{symbol}.zip ---
    def save(self, repo_root: str | Path = ".", extra: Dict | None = None):
        root = Path(repo_root)
        out_dir = root / "PPO_models"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"PPO_{self.cfg.symbol}.zip"

        # Serializamos state_dict y config
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)

        meta = {
            "config": self.cfg.__dict__,
            "extra": extra or {},
        }
        meta_bytes = json.dumps(meta).encode("utf-8")

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pt", buffer.read())
            zf.writestr("meta.json", meta_bytes)

        return out_path

    def load(self, repo_root: str | Path = "."):
        root = Path(repo_root)
        in_path = root / "PPO_models" / f"PPO_{self.cfg.symbol}.zip"
        if not in_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {in_path}")

        with zipfile.ZipFile(in_path, "r") as zf:
            with zf.open("model.pt", "r") as f:
                state_bytes = f.read()
            buf = io.BytesIO(state_bytes)
            state = torch.load(buf, map_location=self.device)
            self.model.load_state_dict(state)
        return True


# ============== Stub mínimo si no hay Torch ==============
class PPOAgentStub:
    """
    Agente stub (sin torch): actúa aleatoriamente y guarda solo metadatos.
    Útil para tests de cableado. Para entrenar de verdad, instala torch.
    """
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def act(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        action = int(np.random.randint(0, 3))
        return action, 0.0, 0.0

    def update(self, batch: Dict) -> Dict:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}

    def save(self, repo_root: str | Path = ".", extra: Dict | None = None):
        root = Path(repo_root)
        out_dir = root / "PPO_models"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"PPO_{self.cfg.symbol}.zip"
        meta = {"config": self.cfg.__dict__, "extra": extra or {}, "note": "stub-without-torch"}
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta).encode("utf-8"))
        return out_path

    def load(self, repo_root: str | Path = "."):
        in_path = Path(repo_root) / "PPO_models" / f"PPO_{self.cfg.symbol}.zip"
        return in_path.exists()


# ============== Fábrica pública ==============
def make_ppo_agent(cfg: PPOConfig):
    if _HAS_TORCH:
        return PPOAgentTorch(cfg)
    return PPOAgentStub(cfg)
