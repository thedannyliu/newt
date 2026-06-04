import argparse
import csv
import json
from pathlib import Path


CELLS = [
	("mlp", "gaussian"),
	("flow", "gaussian"),
	("mlp", "flow"),
	("flow", "flow"),
]


def read_last_eval(metrics_fp: Path):
	last = None
	with metrics_fp.open() as f:
		for line in f:
			row = json.loads(line)
			if row.get("category") == "eval":
				last = row
	return last


def main():
	parser = argparse.ArgumentParser(description="Aggregate local Newt flow 2x2 metrics.jsonl files.")
	parser.add_argument("--logs-root", default="./outputs/logs/soup/1", help="Root containing experiment directories.")
	parser.add_argument("--prefix", default="wm-", help="Experiment prefix used by formal runs.")
	parser.add_argument("--out", default="./csv/flow_2x2_avg.csv", help="Output CSV path.")
	args = parser.parse_args()

	logs_root = Path(args.logs_root)
	rows = []
	for dynamics, policy in CELLS:
		exp_name = f"{args.prefix}{dynamics}_pi-{policy}_seed-1"
		metrics_fp = logs_root / exp_name / "metrics.jsonl"
		if not metrics_fp.exists():
			rows.append({
				"dynamics_arch": dynamics,
				"policy_arch": policy,
				"status": "missing",
			})
			continue
		row = read_last_eval(metrics_fp)
		if row is None:
			rows.append({
				"dynamics_arch": dynamics,
				"policy_arch": policy,
				"status": "no_eval",
			})
			continue
		rows.append({
			"dynamics_arch": dynamics,
			"policy_arch": policy,
			"status": "ok",
			"step": row.get("step"),
			"episode_score": row.get("episode_score"),
			"avg_score": row.get("avg_score"),
			"avg_score_weighted": row.get("avg_score_weighted"),
			"episode_success": row.get("episode_success"),
			"episode_reward": row.get("episode_reward"),
		})

	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"dynamics_arch",
		"policy_arch",
		"status",
		"step",
		"episode_score",
		"avg_score",
		"avg_score_weighted",
		"episode_success",
		"episode_reward",
	]
	with out.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	print(out)


if __name__ == "__main__":
	main()
