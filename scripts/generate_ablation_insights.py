#!/usr/bin/env python3
"""Generate professor-facing ablation tables and figures from local metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/logs/soup/1")


def read_jsonl(path: Path) -> list[dict]:
	records = []
	with path.open() as f:
		for line in f:
			line = line.strip()
			if line:
				records.append(json.loads(line))
	return records


def mean(xs: list[float]) -> float:
	return sum(xs) / len(xs) if xs else float("nan")


def std(xs: list[float]) -> float:
	if len(xs) < 2:
		return 0.0
	m = mean(xs)
	return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x: float | int | None, digits: int = 3) -> str:
	if x is None or (isinstance(x, float) and math.isnan(x)):
		return ""
	if isinstance(x, int):
		return str(x)
	return f"{x:.{digits}f}"


def parse_subset40_name(name: str) -> dict | None:
	m = re.match(
		r"subset40-5m-wmvar-(?P<gpu>.+?)_wm-(?P<wm>.+)_pi-(?P<policy>.+)_seed-(?P<seed>\d+)$",
		name,
	)
	if m:
		return {"family": "subset40_wmvar", **m.groupdict()}
	m = re.match(r"subset40-5m-(?P<gpu>[^_]+)_wm-(?P<wm>.+)_pi-(?P<policy>.+)_seed-(?P<seed>\d+)$", name)
	if m:
		return {"family": "subset40_2x2", **m.groupdict()}
	return None


def parse_eval_subset40_name(name: str) -> dict | None:
	# Strip the eval timestamp suffix.
	name = re.sub(r"-20\d{6}[a-z]$", "", name)
	if not name.startswith("eval-"):
		return None
	return parse_subset40_name(name.removeprefix("eval-"))


def parse_subset20_name(name: str) -> dict | None:
	m = re.match(
		r"subset20-10m-(?P<gpu>[^_]+)_wm-(?P<wm>.+)_pi-(?P<policy>.+)_seed-(?P<seed>\d+)$",
		name,
	)
	if m:
		return {"family": "subset20_10m", **m.groupdict()}
	return None


def dedup_eval_curve(records: list[dict]) -> list[dict]:
	by_step = {}
	for rec in records:
		if rec.get("category") != "eval":
			continue
		step = int(rec.get("step", -1))
		score = rec.get("avg_score", rec.get("episode_score"))
		if step < 0 or score is None:
			continue
		by_step[step] = rec
	return [by_step[s] for s in sorted(by_step)]


def run_summary(path: Path, meta: dict) -> dict:
	records = read_jsonl(path / "metrics.jsonl")
	evals = dedup_eval_curve(records)
	trains = [r for r in records if r.get("category") == "train" and "step" in r]
	latest_eval = max(evals, key=lambda r: r.get("step", -1), default={})
	best_eval = max(evals, key=lambda r: r.get("avg_score", r.get("episode_score", -1)), default={})
	max_train = max(trains, key=lambda r: r.get("step", -1), default={})
	return {
		**meta,
		"name": path.name,
		"path": str(path),
		"train_step": int(max_train.get("step", 0) or 0),
		"eval_step": int(latest_eval.get("step", 0) or 0),
		"final_score": latest_eval.get("avg_score", latest_eval.get("episode_score")),
		"peak_step": int(best_eval.get("step", 0) or 0),
		"peak_score": best_eval.get("avg_score", best_eval.get("episode_score")),
		"action_time": max_train.get("action_time"),
		"update_time": max_train.get("update_time"),
		"steps_per_second": max_train.get("steps_per_second"),
		"elapsed_time_hours": (max_train.get("elapsed_time") / 3600.0) if max_train.get("elapsed_time") else None,
		"curve": [(int(r["step"]), float(r.get("avg_score", r.get("episode_score")))) for r in evals],
	}


def collect_runs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
	subset40_train, subset40_eval, subset20, flowsteps = [], [], [], []
	for path in sorted(ROOT.glob("*/metrics.jsonl")):
		run_dir = path.parent
		name = run_dir.name
		if name.startswith("eval-subset40-5m"):
			meta = parse_eval_subset40_name(name)
			if meta:
				subset40_eval.append(run_summary(run_dir, meta))
			continue
		meta40 = parse_subset40_name(name)
		if meta40:
			subset40_train.append(run_summary(run_dir, meta40))
			continue
		meta20 = parse_subset20_name(name)
		if meta20:
			subset20.append(run_summary(run_dir, meta20))
			continue
		m = re.match(r"subset40-flowsteps(?P<flow_steps>\d+)-(?P<gpu>[^_]+)_wm-(?P<wm>.+)_pi-(?P<policy>.+)_seed-(?P<seed>\d+)$", name)
		if m:
			flowsteps.append(run_summary(run_dir, {"family": "subset40_flowsteps", **m.groupdict()}))
	return subset40_train, subset40_eval, subset20, flowsteps


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fields)
		writer.writeheader()
		for row in rows:
			writer.writerow({k: row.get(k, "") for k in fields})


def summarize_group(rows: list[dict], key_fields: list[str]) -> list[dict]:
	groups = defaultdict(list)
	for row in rows:
		if row.get("final_score") is None:
			continue
		key = tuple(row.get(k) for k in key_fields)
		groups[key].append(row)
	out = []
	for key, vals in sorted(groups.items()):
		finals = [float(v["final_score"]) for v in vals if v.get("final_score") is not None]
		peaks = [float(v["peak_score"]) for v in vals if v.get("peak_score") is not None]
		sps = [float(v["steps_per_second"]) for v in vals if v.get("steps_per_second")]
		actions = [float(v["action_time"]) for v in vals if v.get("action_time")]
		updates = [float(v["update_time"]) for v in vals if v.get("update_time")]
		rec = {field: key[i] for i, field in enumerate(key_fields)}
		rec.update(
			n=len(vals),
			final_mean=mean(finals),
			final_std=std(finals),
			peak_max=max(peaks) if peaks else None,
			peak_mean=mean(peaks),
			action_time_mean=mean(actions),
			update_time_mean=mean(updates),
			sps_mean=mean(sps),
			est_hours_to_10m=(10_000_000 / mean(sps) / 3600.0) if sps and mean(sps) > 0 else None,
			best_run=max(vals, key=lambda v: v.get("peak_score") or -1).get("name"),
		)
		out.append(rec)
	return out


def plot_2x2_heatmap(rows: list[dict], out: Path) -> None:
	cells = {(r["wm"], r["policy"]): r for r in rows if r["wm"] in {"mlp", "flow"}}
	wms = ["mlp", "flow"]
	pols = ["gaussian", "flow"]
	data = [[cells.get((wm, pol), {}).get("final_mean", float("nan")) for pol in pols] for wm in wms]
	fig, ax = plt.subplots(figsize=(6.5, 4.6))
	im = ax.imshow(data, cmap="viridis", vmin=0, vmax=max(max(row) for row in data if row))
	ax.set_xticks(range(len(pols)), pols)
	ax.set_yticks(range(len(wms)), wms)
	ax.set_xlabel("Policy prior")
	ax.set_ylabel("World model dynamics")
	ax.set_title("Subset40 5M WM x Policy 2x2: final avg_score")
	for i, wm in enumerate(wms):
		for j, pol in enumerate(pols):
			rec = cells.get((wm, pol))
			if rec:
				ax.text(j, i, f"{rec['final_mean']:.3f}\n±{rec['final_std']:.3f}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
	fig.colorbar(im, ax=ax, label="avg_score")
	fig.tight_layout()
	fig.savefig(out, dpi=180)
	plt.close(fig)


def plot_bar(rows: list[dict], out: Path, title: str, ylabel: str = "avg_score") -> None:
	rows = sorted(rows, key=lambda r: r.get("final_mean", 0), reverse=True)
	labels = [r["wm"].replace("_", "\n") for r in rows]
	vals = [r["final_mean"] for r in rows]
	errs = [r["final_std"] for r in rows]
	fig, ax = plt.subplots(figsize=(10, 5.2))
	ax.bar(range(len(rows)), vals, yerr=errs, capsize=4, color="#5176b8")
	ax.set_xticks(range(len(rows)), labels, fontsize=8)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(axis="y", alpha=0.25)
	for i, r in enumerate(rows):
		ax.text(i, vals[i] + 0.008, f"peak {r['peak_max']:.3f}", ha="center", fontsize=8)
	fig.tight_layout()
	fig.savefig(out, dpi=180)
	plt.close(fig)


def plot_curves(grouped: dict[str, list[dict]], out: Path, title: str) -> None:
	fig, ax = plt.subplots(figsize=(9, 5.2))
	for label, runs in grouped.items():
		by_step = defaultdict(list)
		for run in runs:
			for step, score in run["curve"]:
				by_step[step].append(score)
		if not by_step:
			continue
		xs = sorted(by_step)
		ys = [mean(by_step[x]) for x in xs]
		ax.plot([x / 1e6 for x in xs], ys, marker="o", linewidth=2, label=label)
	ax.set_xlabel("Environment steps (M)")
	ax.set_ylabel("eval avg_score")
	ax.set_title(title)
	ax.grid(alpha=0.25)
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(out, dpi=180)
	plt.close(fig)


def plot_compute_scatter(rows: list[dict], out: Path) -> None:
	fig, ax = plt.subplots(figsize=(8, 5.4))
	color_map = {
		"mlp+gaussian": "#4c78a8",
		"mlp+flow": "#f58518",
		"residual_flow+gaussian": "#54a24b",
		"residual_mean_flow_wm+gaussian": "#b279a2",
		"flow+gaussian": "#e45756",
		"flow+flow": "#72b7b2",
	}
	for row in rows:
		if not row.get("action_time") or not row.get("peak_score"):
			continue
		label = f"{row['wm']}+{row['policy']}"
		ax.scatter(
			float(row["action_time"]),
			float(row["peak_score"]),
			s=70,
			alpha=0.85,
			color=color_map.get(label, "#777777"),
			label=label,
		)
		ax.text(float(row["action_time"]) + 0.001, float(row["peak_score"]), row["gpu"].split("-")[0], fontsize=7)
	handles, labels = ax.get_legend_handles_labels()
	uniq = dict(zip(labels, handles))
	ax.legend(uniq.values(), uniq.keys(), fontsize=8)
	ax.set_xlabel("action_time (s, latest train log)")
	ax.set_ylabel("peak eval avg_score")
	ax.set_title("Compute vs Performance: subset20 high-step runs")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(out, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--out-dir", default="docs/assets/ablation_insights_20260610")
	args = parser.parse_args()
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	subset40_train, subset40_eval, subset20, flowsteps = collect_runs()
	subset40_2x2_eval = [r for r in subset40_eval if r["family"] == "subset40_2x2"]
	subset40_wmvar_eval = [r for r in subset40_eval if r["family"] == "subset40_wmvar"]

	# Use direct eval rows for score summaries; training rows for runtime/curves.
	two_by_two = summarize_group(subset40_2x2_eval, ["wm", "policy"])
	wm_variants = summarize_group(
		[r for r in subset40_wmvar_eval + [r for r in subset40_2x2_eval if r["wm"] == "flow" and r["policy"] == "gaussian"] if r["policy"] == "gaussian"],
		["wm", "policy"],
	)
	subset20_summary = summarize_group(subset20, ["wm", "policy"])
	flowsteps_summary = summarize_group(flowsteps, ["flow_steps", "wm", "policy"])

	write_csv(out_dir / "subset40_2x2_summary.csv", two_by_two, ["wm", "policy", "n", "final_mean", "final_std", "peak_max", "peak_mean", "best_run"])
	write_csv(out_dir / "subset40_flow_wm_variants_summary.csv", wm_variants, ["wm", "policy", "n", "final_mean", "final_std", "peak_max", "peak_mean", "best_run"])
	write_csv(out_dir / "subset20_highstep_summary.csv", subset20_summary, ["wm", "policy", "n", "final_mean", "final_std", "peak_max", "peak_mean", "action_time_mean", "update_time_mean", "sps_mean", "est_hours_to_10m", "best_run"])
	write_csv(out_dir / "subset20_runs.csv", subset20, ["gpu", "wm", "policy", "train_step", "eval_step", "final_score", "peak_step", "peak_score", "action_time", "update_time", "steps_per_second", "elapsed_time_hours", "name", "path"])
	write_csv(out_dir / "subset40_flowsteps_summary.csv", flowsteps_summary, ["flow_steps", "wm", "policy", "n", "final_mean", "final_std", "peak_max", "action_time_mean", "update_time_mean", "sps_mean", "best_run"])

	plot_2x2_heatmap(two_by_two, out_dir / "subset40_2x2_heatmap.png")
	plot_bar(wm_variants, out_dir / "subset40_flow_wm_variants.png", "Subset40 5M Flow-WM variants: final avg_score")

	curve_groups_40 = defaultdict(list)
	for run in subset40_train:
		if run["family"] == "subset40_2x2":
			curve_groups_40[f"{run['wm']}+{run['policy']}"].append(run)
	plot_curves(curve_groups_40, out_dir / "subset40_2x2_training_curves.png", "Subset40 5M training curves by architecture")

	keep20 = {
		("mlp", "gaussian"),
		("mlp", "flow"),
		("residual_flow", "gaussian"),
		("residual_mean_flow_wm", "gaussian"),
	}
	curve_groups_20 = defaultdict(list)
	for run in subset20:
		if (run["wm"], run["policy"]) in keep20:
			curve_groups_20[f"{run['wm']}+{run['policy']}"].append(run)
	plot_curves(curve_groups_20, out_dir / "subset20_highstep_training_curves.png", "Subset20 10M high-step training curves")
	plot_compute_scatter(subset20, out_dir / "subset20_compute_vs_score.png")

	scaling_rows = [
		{"setting": "subset40 5M", "tasks": 40, "steps_m": 5, "steps_per_task_k": 125, "score": max(r["final_mean"] for r in two_by_two)},
		{"setting": "subset20 10M best completed", "tasks": 20, "steps_m": 10, "steps_per_task_k": 500, "score": max(r["final_score"] for r in subset20 if r["eval_step"] >= 10_000_000)},
		{"setting": "official 200 20M", "tasks": 200, "steps_m": 20, "steps_per_task_k": 100, "score": 0.31},
		{"setting": "official 200 100M", "tasks": 200, "steps_m": 100, "steps_per_task_k": 500, "score": 0.438},
	]
	write_csv(out_dir / "scaling_summary.csv", scaling_rows, ["setting", "tasks", "steps_m", "steps_per_task_k", "score"])
	fig, ax = plt.subplots(figsize=(7, 4.8))
	for row in scaling_rows:
		ax.scatter(row["steps_per_task_k"], row["score"], s=90)
		ax.text(row["steps_per_task_k"] + 5, row["score"], row["setting"], fontsize=8)
	ax.set_xlabel("Environment steps per task (K)")
	ax.set_ylabel("avg_score")
	ax.set_title("Task/step scaling signal")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_dir / "scaling_steps_per_task.png", dpi=180)
	plt.close(fig)

	print(f"Wrote ablation assets to {out_dir}")


if __name__ == "__main__":
	main()
