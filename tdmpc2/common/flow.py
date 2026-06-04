import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from common import layers


class TimeEmbedding(nn.Module):
	"""Sinusoidal time embedding for scalar flow times."""

	def __init__(self, dim: int):
		super().__init__()
		assert dim % 2 == 0, "flow_t_dim must be even"
		self.dim = dim

	def forward(self, t):
		t = t.to(dtype=torch.float32)
		half = self.dim // 2
		freq = torch.exp(
			torch.arange(half, device=t.device, dtype=t.dtype)
			* (-torch.log(torch.tensor(10_000.0, device=t.device, dtype=t.dtype)) / max(half - 1, 1))
		)
		x = t * freq
		return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ConditionalFlow(nn.Module):
	"""
	Small rectified-flow velocity model.
	Inputs are a noisy sample x_t, scalar time t, and a conditioning vector.
	"""

	def __init__(self, cfg, sample_dim: int, cond_dim: int, out_act=None):
		super().__init__()
		self.sample_dim = sample_dim
		self.t_embed = TimeEmbedding(cfg.flow_t_dim)
		self.net = layers.mlp(
			sample_dim + cond_dim + cfg.flow_t_dim,
			cfg.flow_hidden_layers * [cfg.mlp_dim],
			sample_dim,
			act=out_act,
		)

	def velocity(self, x, cond, t):
		if t.ndim == x.ndim - 1:
			t = t.unsqueeze(-1)
		t_emb = self.t_embed(t)
		return self.net(torch.cat([x, cond, t_emb], dim=-1))

	def loss(self, target, cond, mask=None):
		noise = torch.randn_like(target)
		t = torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		x_t = (1 - t) * noise + t * target
		target_velocity = target - noise
		pred_velocity = self.velocity(x_t, cond, t)
		loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
		if mask is not None:
			while mask.ndim < loss.ndim:
				mask = mask.unsqueeze(0)
			mask = mask.expand_as(loss)
			return (loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
		return loss.mean(-1)

	def sample(self, cond, base=None, steps: int = 4, clamp: bool = False):
		if base is None:
			base = torch.randn(*cond.shape[:-1], self.sample_dim, device=cond.device, dtype=cond.dtype)
		x = base
		dt = 1.0 / steps
		for i in range(steps):
			t = torch.full((*x.shape[:-1], 1), fill_value=(i + 0.5) * dt, device=x.device, dtype=x.dtype)
			x = x + dt * self.velocity(x, cond, t)
			if clamp:
				x = x.clamp(-1, 1)
		return x


class FlowPolicy(nn.Module):
	"""Conditional action flow used as Newt's policy prior."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.flow = ConditionalFlow(cfg, cfg.action_dim, cfg.latent_dim + cfg.task_dim)

	def forward(self, z, action_mask, deterministic=False):
		base = torch.zeros(*z.shape[:-1], self.cfg.action_dim, device=z.device, dtype=z.dtype) if deterministic else None
		action = self.flow.sample(z, base=base, steps=self.cfg.flow_steps, clamp=True)
		while action_mask.ndim < action.ndim:
			action_mask = action_mask.unsqueeze(-2)
		action_mask = action_mask.expand_as(action)
		action = action * action_mask
		info = TensorDict({
			"mean": self.flow.sample(
				z,
				base=torch.zeros_like(action),
				steps=self.cfg.flow_steps,
				clamp=True,
			) * action_mask,
			"log_std": torch.zeros_like(action),
			"entropy": torch.zeros(*action.shape[:-1], 1, device=action.device, dtype=action.dtype),
			"scaled_entropy": torch.zeros(*action.shape[:-1], 1, device=action.device, dtype=action.dtype),
		})
		return action, info

	def loss(self, z, action, action_mask):
		return self.flow.loss(action, z, mask=action_mask)
