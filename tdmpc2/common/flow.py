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


class OTConditionalFlow(ConditionalFlow):
	"""
	Rectified flow with a lightweight sliced-OT source/target pairing.
	Target conditioning stays attached to each batch row; only the Gaussian
	source samples are reordered to make the source-target coupling straighter.
	"""

	def loss(self, target, cond, mask=None):
		assert mask is None, "OTConditionalFlow is only used for dynamics residuals"
		noise = torch.randn_like(target)
		target_flat = target.reshape(-1, target.shape[-1])
		noise_flat = noise.reshape(-1, noise.shape[-1])
		proj = torch.randn(target_flat.shape[-1], device=target.device, dtype=target.dtype)
		proj = proj / proj.norm().clamp_min(1e-6)
		target_order = (target_flat @ proj).argsort()
		noise_order = (noise_flat @ proj).argsort()
		paired_noise = noise_flat.clone()
		paired_noise[target_order] = noise_flat[noise_order]
		noise = paired_noise.reshape_as(target)
		t = torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		x_t = (1 - t) * noise + t * target
		target_velocity = target - noise
		pred_velocity = self.velocity(x_t, cond, t)
		return F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(-1)


class MeanFlow(nn.Module):
	"""
	One-step average-velocity flow.
	This is a compact latent residual pilot inspired by MeanFlow: the network
	predicts average velocity over an interval [r, t], enabling one function
	evaluation at rollout time.
	"""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__()
		self.sample_dim = sample_dim
		self.t_embed = TimeEmbedding(cfg.flow_t_dim)
		self.net = layers.mlp(
			sample_dim + cond_dim + 2 * cfg.flow_t_dim,
			cfg.flow_hidden_layers * [cfg.mlp_dim],
			sample_dim,
		)

	def average_velocity(self, x, cond, r, t):
		if r.ndim == x.ndim - 1:
			r = r.unsqueeze(-1)
		if t.ndim == x.ndim - 1:
			t = t.unsqueeze(-1)
		return self.net(torch.cat([x, cond, self.t_embed(r), self.t_embed(t)], dim=-1))

	def loss(self, target, cond):
		noise = torch.randn_like(target)
		r = torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		t = r + (1 - r) * torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		x_r = (1 - r) * noise + r * target
		target_velocity = target - noise
		pred_velocity = self.average_velocity(x_r, cond, r, t)
		return F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(-1)

	def sample(self, cond, base=None):
		if base is None:
			base = torch.zeros(*cond.shape[:-1], self.sample_dim, device=cond.device, dtype=cond.dtype)
		r = torch.zeros(*base.shape[:-1], 1, device=base.device, dtype=base.dtype)
		t = torch.ones(*base.shape[:-1], 1, device=base.device, dtype=base.dtype)
		return base + self.average_velocity(base, cond, r, t)


class ShortcutFlow(nn.Module):
	"""
	Step-size-conditioned flow for few-step or one-step residual rollouts.
	"""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__()
		self.sample_dim = sample_dim
		self.t_embed = TimeEmbedding(cfg.flow_t_dim)
		self.dt_embed = TimeEmbedding(cfg.flow_t_dim)
		self.net = layers.mlp(
			sample_dim + cond_dim + 2 * cfg.flow_t_dim,
			cfg.flow_hidden_layers * [cfg.mlp_dim],
			sample_dim,
		)

	def velocity(self, x, cond, t, dt):
		if t.ndim == x.ndim - 1:
			t = t.unsqueeze(-1)
		if dt.ndim == x.ndim - 1:
			dt = dt.unsqueeze(-1)
		return self.net(torch.cat([x, cond, self.t_embed(t), self.dt_embed(dt)], dim=-1))

	def loss(self, target, cond):
		noise = torch.randn_like(target)
		t = torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		max_dt = (1 - t).clamp_min(1e-4)
		dt = max_dt * torch.rand(*target.shape[:-1], 1, device=target.device, dtype=target.dtype)
		x_t = (1 - t) * noise + t * target
		target_velocity = target - noise
		pred_velocity = self.velocity(x_t, cond, t, dt)
		return F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(-1)

	def sample(self, cond, base=None, steps: int = 4):
		if base is None:
			base = torch.zeros(*cond.shape[:-1], self.sample_dim, device=cond.device, dtype=cond.dtype)
		x = base
		steps = max(int(steps), 1)
		dt_value = 1.0 / steps
		for i in range(steps):
			t = torch.full((*x.shape[:-1], 1), fill_value=i * dt_value, device=x.device, dtype=x.dtype)
			dt = torch.full_like(t, dt_value)
			x = x + dt * self.velocity(x, cond, t, dt)
		return x


class EndpointFlowDynamics(nn.Module):
	"""
	Deterministic endpoint residual predictor.
	This keeps the flow-style time-conditioned interface but predicts the
	latent residual in one network call instead of Euler sampling.
	"""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__()
		self.sample_dim = sample_dim
		self.t_embed = TimeEmbedding(cfg.flow_t_dim)
		self.net = layers.mlp(
			sample_dim + cond_dim + cfg.flow_t_dim,
			cfg.flow_hidden_layers * [cfg.mlp_dim],
			sample_dim,
		)

	def forward(self, cond):
		base = torch.zeros(*cond.shape[:-1], self.sample_dim, device=cond.device, dtype=cond.dtype)
		t = torch.ones(*cond.shape[:-1], 1, device=cond.device, dtype=cond.dtype)
		t_emb = self.t_embed(t)
		return self.net(torch.cat([base, cond, t_emb], dim=-1))

	def loss(self, target, cond):
		return F.mse_loss(self.forward(cond), target, reduction='none').mean(-1)


class ResidualFlowDynamics(nn.Module):
	"""
	MLP dynamics with a flow residual correction.
	The MLP path learns the stable baseline next latent; the flow path models
	the remaining residual without replacing the baseline dynamics.
	"""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__()
		self.cfg = cfg
		self.base = layers.mlp(
			cond_dim,
			2 * [cfg.mlp_dim],
			sample_dim,
			act=layers.SimNorm(cfg),
		)
		self.flow = ConditionalFlow(cfg, sample_dim, cond_dim)

	def forward(self, cond, steps: int = 4):
		base = self.base(cond)
		residual = self.flow.sample(cond, base=torch.zeros_like(base), steps=steps)
		return base, residual

	def loss(self, target_z, cond):
		base = self.base(cond)
		base_loss = F.mse_loss(base, target_z, reduction='none').mean(-1)
		residual_target = (target_z - base).detach()
		flow_loss = self.flow.loss(residual_target, cond)
		return base_loss + flow_loss


class ResidualMeanFlowDynamics(ResidualFlowDynamics):
	"""MLP next-latent dynamics plus one-step mean-flow residual correction."""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__(cfg, sample_dim, cond_dim)
		self.flow = MeanFlow(cfg, sample_dim, cond_dim)

	def forward(self, cond, steps: int = 4):
		base = self.base(cond)
		residual = self.flow.sample(cond, base=torch.zeros_like(base))
		return base, residual


class ShortcutResidualFlowDynamics(ResidualFlowDynamics):
	"""MLP next-latent dynamics plus step-size-conditioned shortcut residual flow."""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__(cfg, sample_dim, cond_dim)
		self.flow = ShortcutFlow(cfg, sample_dim, cond_dim)


class OTResidualFlowDynamics(ResidualFlowDynamics):
	"""MLP next-latent dynamics plus sliced-OT rectified-flow residual correction."""

	def __init__(self, cfg, sample_dim: int, cond_dim: int):
		super().__init__(cfg, sample_dim, cond_dim)
		self.flow = OTConditionalFlow(cfg, sample_dim, cond_dim)


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
