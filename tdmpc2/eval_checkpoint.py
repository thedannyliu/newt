import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
import warnings
warnings.filterwarnings('ignore')

import torch
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	assert cfg.checkpoint is not None, "checkpoint is required for eval_checkpoint.py"
	if not os.path.exists(cfg.checkpoint):
		raise FileNotFoundError(f"Checkpoint file not found: {cfg.checkpoint}")
	cfg = parse_cfg(cfg)
	cfg.rank = 0
	cfg.world_size = 1

	env = make_env(cfg)
	torch.cuda.set_device(0)
	set_seed(cfg.seed)
	model = WorldModel(cfg).to("cuda:0")
	agent = TDMPC2(model, cfg)
	logger = Logger(cfg)
	trainer = Trainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=None,
		logger=logger,
	)
	checkpoint_state = trainer.agent.load(cfg.checkpoint, load_training_state=False)
	if isinstance(checkpoint_state, dict):
		if "scale" in checkpoint_state:
			trainer.agent.scale.load_state_dict(checkpoint_state["scale"])
		extra_state = checkpoint_state.get("extra", {})
		if isinstance(extra_state, dict) and "trainer" in extra_state:
			trainer.load_state_dict({"trainer": extra_state["trainer"]})
	print(colored(f"Loaded checkpoint from {cfg.checkpoint}.", "blue", attrs=["bold"]))

	eval_metrics = trainer.eval()
	eval_metrics.update(trainer.common_metrics())
	if cfg.task == "soup":
		logger.pprint_multitask(eval_metrics, cfg)
	logger.log(eval_metrics, "eval")
	logger.finish()
	print(colored("Evaluation completed successfully", "green", attrs=["bold"]))


if __name__ == '__main__':
	launch()
