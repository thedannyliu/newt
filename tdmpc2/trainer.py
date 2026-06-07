import os
import random
from collections import defaultdict, OrderedDict
from time import time

import torch
import numpy as np
from termcolor import colored
from tqdm import tqdm
from tensordict.tensordict import TensorDict

from common import barrier


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def empty_metrics():
	return {'reward': [], 'length': [], 'success': [], 'score': []}


class Trainer():
	"""Trainer class for MMBench experiments."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._tasks = torch.tensor(split_by_rank(range(cfg.num_global_tasks), cfg.rank, cfg.world_size),
			dtype=torch.int32, device='cpu')
		self._episode_lengths = torch.tensor(split_by_rank(cfg.episode_lengths, cfg.rank, cfg.world_size),
			dtype=torch.int32, device='cpu')
		if cfg.task != 'soup':
			self._tasks = self._tasks.repeat_interleave(cfg.num_envs)
		assert self.cfg.episode_length in self._episode_lengths, \
			f'[Rank {cfg.rank}] Expected maximum episode length {self.cfg.episode_length} to be in {self._episode_lengths.tolist()}.'
		self._tds = TensorDict({}, batch_size=(self.cfg.episode_length+1, self.cfg.num_envs), device='cpu')
		self._update_freq = self.cfg.num_envs * self.cfg.episode_length * self.cfg.world_size
		self._update_tokens = 0
		self._last_checkpoint_step = 0
		self._last_replay_checkpoint_step = 0
		self._episode_boundary_offset = 0
		self._eps_per_update_freq = int((cfg.episode_length / np.array(cfg.episode_lengths)).sum())
		if cfg.rank == 0:
			print('Architecture:', self.agent.model)
			print(f'Update frequency: {self._update_freq:,}')
			print(f'Episodes per update frequency: {self._eps_per_update_freq:,}')

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def state_dict(self, include_replay=False):
		"""Return state required for interrupted training to resume."""
		state = {
			"trainer": {
				"step": self._step,
				"episode": self._ep_idx,
				"update_tokens": self._update_tokens,
				"last_checkpoint_step": self._last_checkpoint_step,
				"last_replay_checkpoint_step": self._last_replay_checkpoint_step,
				"episode_boundary_offset": self._episode_boundary_offset,
			},
			"rng": {
				"python": random.getstate(),
				"numpy": np.random.get_state(),
				"torch": torch.get_rng_state(),
				"cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
			},
		}
		if include_replay and self.cfg.save_replay:
			state["replay"] = self.buffer.state_dict()
		return state

	def load_state_dict(self, state):
		"""Restore trainer, RNG, and optional replay state."""
		trainer_state = state.get("trainer", {})
		self._step = trainer_state.get("step", self._step)
		self._ep_idx = trainer_state.get("episode", self._ep_idx)
		self._update_tokens = trainer_state.get("update_tokens", self._update_tokens)
		self._last_checkpoint_step = trainer_state.get("last_checkpoint_step", self._last_checkpoint_step)
		self._last_replay_checkpoint_step = trainer_state.get("last_replay_checkpoint_step", self._last_replay_checkpoint_step)
		# Checkpoints do not store the partially collected episode tensor or env state.
		# After reset, the next full vectorized episode boundary is offset by the
		# checkpoint step, even when an older checkpoint stored offset 0.
		self._episode_boundary_offset = self._step % self._update_freq
		rng_state = state.get("rng", {})
		if "python" in rng_state:
			random.setstate(rng_state["python"])
		if "numpy" in rng_state:
			np.random.set_state(rng_state["numpy"])
		if "torch" in rng_state:
			torch.set_rng_state(rng_state["torch"].cpu())
		if rng_state.get("cuda") is not None and torch.cuda.is_available():
			torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])
		if "replay" in state:
			self.buffer.load_state_dict(state["replay"])

	def save_checkpoint(self, identifier, include_replay=False, upload_artifact=True):
		"""Save a model checkpoint with optional full resume state."""
		extra_state = self.state_dict(include_replay=include_replay)
		self.logger.save_agent(
			self.agent,
			identifier,
			extra_state=extra_state,
			upload_artifact=upload_artifact and not include_replay,
		)

	def eval(self):
		"""Evaluate agent and aggregate all completed episodes per unique task name."""
		task_results = defaultdict(empty_metrics)

		obs, info = self.env.reset()
		episode_reward = torch.zeros(self.cfg.num_envs)
		episode_len = torch.zeros(self.cfg.num_envs)
		episodes_completed = torch.zeros(self.cfg.num_envs, dtype=torch.int32)

		if self.cfg.save_video:
			self.logger.video.init(self.env, enabled=self.cfg.rank==0)

		while (episodes_completed < self.cfg.eval_episodes).any():
			use_mpc = self._step > 0 or self.cfg.finetune
			torch.compiler.cudagraph_mark_step_begin()
			action = self.agent(obs, t0=episode_len==0, step=self._step, eval_mode=True, task=self._tasks, mpc=use_mpc)
			obs, reward, terminated, truncated, info = self.env.step(action)

			done = terminated | truncated
			episode_reward += reward
			episode_len += 1

			if 'final_info' in info:
				for i in range(self.cfg.num_envs):
					if done[i]:
						task_id = self._tasks[i].item()
						task_name = self.cfg.global_tasks[task_id]

						task_results[task_name]['reward'].append(episode_reward[i].item())
						task_results[task_name]['length'].append(episode_len[i].item())
						task_results[task_name]['success'].append(info['final_info']['success'][i].item())
						task_results[task_name]['score'].append(info['final_info']['score'][i].item())

						episode_reward[i] = 0.0
						episode_len[i] = 0.0
						episodes_completed[i] += 1

			if self.cfg.save_video and episodes_completed.min() == 0:
				self.logger.video.record(self.env)

		if self.cfg.save_video:
			self.logger.video.save(self._step)

		barrier()  # Ensure all processes have completed evaluation

		if self.cfg.world_size > 1:
			# Gather results from all ranks
			gathered_results = [None for _ in range(self.cfg.world_size)] if self.cfg.rank == 0 else None
			torch.distributed.gather_object(task_results, gathered_results, dst=0)
			if self.cfg.rank == 0:
				# Combine results from all ranks
				for rank_results in gathered_results:
					for task_name, metrics in rank_results.items():
						for metric_name, values in metrics.items():
							task_results[task_name][metric_name].extend(values)

		results = {}
		if self.cfg.task == 'soup' and self.cfg.rank == 0:
			assert len(task_results) == self.cfg.num_global_tasks, \
				f'Expected results for {self.cfg.num_global_tasks} tasks, but got {len(task_results)}.'

		# Sort tasks by order in cfg.global_tasks
		task_results = OrderedDict(sorted(task_results.items(), key=lambda x: self.cfg.global_tasks.index(x[0])))

		# Compute per-task averages
		for task_name, metrics in task_results.items():
			n = len(metrics['reward'])
			results[f'episode_reward+{task_name}'] = sum(metrics['reward']) / n
			results[f'episode_length+{task_name}'] = sum(metrics['length']) / n
			results[f'episode_success+{task_name}'] = sum(metrics['success']) / n
			results[f'episode_score+{task_name}'] = sum(metrics['score']) / n

		# Compute unweighted averages *across tasks*
		num_tasks = len(task_results)
		if self.cfg.rank == 0:
			assert num_tasks == self.cfg.num_global_tasks, \
				f'Number of eval tasks ({num_tasks}) does not match expected ({self.cfg.num_global_tasks})'
		results['episode_reward'] = sum(
			sum(m['reward']) / len(m['reward']) for m in task_results.values()
		) / num_tasks
		results['episode_success'] = sum(
			sum(m['success']) / len(m['success']) for m in task_results.values()
		) / num_tasks
		results['episode_score'] = sum(
			sum(m['score']) / len(m['score']) for m in task_results.values()
		) / num_tasks

		return results

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		if terminated is None:
			terminated = torch.tensor(False).repeat(self.cfg.num_envs)
		elif not isinstance(terminated, torch.Tensor):
			terminated = torch.stack(terminated.tolist())
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
			terminated=terminated,
			task=self._tasks,
			batch_size=(self.cfg.num_envs,))
		return td
	
	def train(self):
		"""Train a Newt agent."""
		# Load demonstrations
		use_demos = self.cfg.get('use_demos', False)
		
		# Load checkpoint
		checkpoint = self.cfg.get('checkpoint', None)
		if checkpoint:
			if not os.path.exists(checkpoint):
				raise FileNotFoundError(f'Checkpoint file not found: {checkpoint}')
			checkpoint_state = self.agent.load(self.cfg.checkpoint)
			if isinstance(checkpoint_state, dict) and "extra" in checkpoint_state:
				self.load_state_dict(checkpoint_state["extra"])
			if self.cfg.rank == 0:
				print(colored(f'Loaded checkpoint from {self.cfg.checkpoint}.', 'blue', attrs=['bold']))
		else:
			checkpoint = None
			if self.cfg.rank == 0:
				print(colored(f'No checkpoint found, training from scratch.', 'yellow', attrs=['bold']))
		
		# Pretrain agent on demonstrations if available
		if use_demos and not checkpoint and self.cfg.demo_steps > 0:
			if self.cfg.rank == 0:
				print('Pretraining agent on demonstrations...')
			self.agent.maxq_pi = False  # Disable max-Q for pretraining
			print(f'prior_coef is {self.agent.cfg.prior_coef}, setting to 1.0 for pretraining.')
			self.agent.cfg.prior_coef = 1.0  # Use only behavior cloning loss
			iterator = tqdm(range(self.cfg.demo_steps), desc='Pretraining') if self.cfg.rank == 0 else range(self.cfg.demo_steps)
			pretrain_log_freq = max(1, int(self.cfg.demo_steps / 50))
			for i in iterator:
				pretrain_metrics = self.agent.update(self.buffer)
				if i % pretrain_log_freq == 0:
					self.logger.pprint_pretrain(pretrain_metrics)
			pretrain_metrics.update({
				'step': 0,
				'elapsed_time': time() - self._start_time,
			})
			self.agent.maxq_pi = True
			self.agent.cfg.prior_coef = self.cfg.prior_coef
			print(f'Set prior_coef to {self.agent.cfg.prior_coef} after pretraining.')
			if self.cfg.rank == 0:
				print('Pretraining complete.')
			self.save_checkpoint(f'{self._step:,}'.replace(',', '_'), include_replay=False)

		# Training loop
		if self.cfg.rank == 0:
			print(f'Training agent for {self.cfg.steps:,} steps...')
		train_metrics = defaultdict(list)
		obs, info = self.env.reset()
		ep_reward = torch.zeros((self.cfg.num_envs,))
		ep_len = torch.zeros((self.cfg.num_envs,), dtype=torch.int32)
		done = torch.full((self.cfg.num_envs,), True, dtype=torch.bool)
		self._next_action = None
		self._tds[ep_len] = self.to_td(obs)
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_metrics = self.eval()
				eval_metrics.update(self.common_metrics())
				if self.cfg.task == 'soup':
					self.logger.pprint_multitask(eval_metrics, self.cfg)
				self.logger.log(eval_metrics, 'eval')

				# Save agent
				if self._step % self.cfg.save_freq == 0 and self._step > 0:
					self.save_checkpoint(f'{self._step:,}'.replace(',', '_'), include_replay=False)

				# Reset environment and metrics
				obs, info = self.env.reset()
				ep_reward = torch.zeros((self.cfg.num_envs,))
				ep_len = torch.zeros((self.cfg.num_envs,), dtype=torch.int32)
				done = torch.full((self.cfg.num_envs,), True, dtype=torch.bool)
				self._next_action = None
				self._tds[ep_len] = self.to_td(obs)

			# Collect experience
			use_mpc = self._step >= self.cfg.seeding_coef * self._update_freq
			use_agent = self.cfg.finetune or (use_demos and self.cfg.demo_steps > 0) or use_mpc
			action_start_time = time()
			if use_agent:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent(obs, t0=done, step=self._step, task=self._tasks, mpc=use_mpc)
			else:
				action = self.env.rand_act()
			train_metrics['action_time'].append(time() - action_start_time)

			env_start_time = time()
			obs, reward, terminated, truncated, info = self.env.step(action)
			train_metrics['env_step_time'].append(time() - env_start_time)
			assert not terminated.any(), \
				f'Unexpected termination signal received.'
			ep_reward += reward
			ep_len += 1
			done = terminated | truncated
			self._step += self.cfg.num_envs * self.cfg.world_size

			# Store experience
			_obs = obs.clone()
			if 'final_observation' in info:
				_obs[done] = info['final_observation']
			td = self.to_td(_obs, action, reward, terminated)
			self._tds[ep_len] = td
			if done.any():
				max_ep_len = ep_len.max()

				for i in range(self.cfg.num_envs):
					if done[i]:
						assert ep_len[i] == self._episode_lengths[i], \
							f'Episode length {ep_len[i]} does not match expected length {self._episode_lengths[i]}.'

						# Add to buffer
						_td = self._tds[:ep_len[i]+1, i].unsqueeze(0)
						self.buffer.add(_td, self.cfg.world_size, self.cfg.rank)

						# Save metrics
						train_metrics['episode_reward'].append(ep_reward[i].item())
						train_metrics['episode_success'].append(info['final_info']['success'][i].item())
						train_metrics['episode_score'].append(info['final_info']['score'][i].item())
						train_metrics['episode_length'].append(ep_len[i].item())
						train_metrics['episode_terminated'].append(terminated[i].item())

						# Reset episode metrics
						ep_reward[i] = 0.0
						ep_len[i] = 0
				
				# Log and reset metrics if enough data is collected
				if max_ep_len >= self.cfg.episode_length:
					current_offset = self._step % self._update_freq
					if current_offset != self._episode_boundary_offset:
						if self.cfg.rank == 0:
							print(
								colored(
									f'Resyncing episode-boundary offset from {self._episode_boundary_offset} '
									f'to {current_offset} at step {self._step}.',
									'yellow',
									attrs=['bold'],
								)
							)
						self._episode_boundary_offset = current_offset
					self._ep_idx += self._eps_per_update_freq
					for key in ['episode_reward', 'episode_success', 'episode_score', 'episode_length', 'episode_terminated']:
						train_metrics[key] = torch.tensor(train_metrics[key], dtype=torch.float32).nanmean().item()
					for key in ['action_time', 'env_step_time', 'update_time', 'num_updates']:
						if len(train_metrics[key]) > 0:
							train_metrics[key] = torch.tensor(train_metrics[key], dtype=torch.float32).nanmean().item()
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					train_metrics = defaultdict(list)
			
			# Update agent
			if self._step >= self.cfg.seeding_coef * self._update_freq:
				self._update_tokens += self.cfg.num_envs * self.cfg.world_size * self.cfg.utd
				if self._update_tokens >= 1.0:
					num_updates = int(self._update_tokens)
					update_start_time = time()
					for _ in range(num_updates):
						_train_metrics = self.agent.update(self.buffer)
					train_metrics['update_time'].append(time() - update_start_time)
					train_metrics['num_updates'].append(num_updates)
					train_metrics.update(_train_metrics)
					self._update_tokens -= num_updates

			if self._step > 0 and self.cfg.save_agent and self._step - self._last_checkpoint_step >= self.cfg.checkpoint_freq:
				self._last_checkpoint_step = self._step
				self.save_checkpoint(f'{self._step:,}'.replace(',', '_'), include_replay=False)
			if self._step > 0 and self.cfg.save_agent and self.cfg.save_replay and \
					self._step - self._last_replay_checkpoint_step >= self.cfg.replay_checkpoint_freq:
				self._last_replay_checkpoint_step = self._step
				self.save_checkpoint(f'{self._step:,}'.replace(',', '_') + '_full', include_replay=True, upload_artifact=False)
		
		self.logger.finish()
