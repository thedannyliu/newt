from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "nicklashansen/mmbench"


def list_demo_files() -> List[str]:
	api = HfApi()
	return sorted([f for f in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset") if f.endswith(".pt")])


def materialize_demo(filename: str, data_dir: Path, cache_dir: Optional[str] = None, token: Optional[str] = None) -> Path:
	if not filename.endswith(".pt"):
		filename = f"{filename}.pt"
	src = Path(hf_hub_download(
		repo_id=REPO_ID,
		repo_type="dataset",
		filename=filename,
		revision="main",
		cache_dir=cache_dir,
		token=token,
	))
	data_dir.mkdir(parents=True, exist_ok=True)
	dst = data_dir / src.name
	if dst.exists():
		return dst
	dst.symlink_to(src)
	return dst


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Download or symlink MMBench demonstration files.")
	parser.add_argument("--all", action="store_true", help="Materialize all MMBench .pt demo files.")
	parser.add_argument("--task", action="append", help="Task/demo filename to materialize. Can be repeated.")
	parser.add_argument("--data-dir", default="./data", help="Directory expected by train.py data_dir.")
	parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
	parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
	args = parser.parse_args()

	tasks = list_demo_files() if args.all else args.task
	if not tasks:
		raise SystemExit("Specify --all or one or more --task values.")

	paths = []
	for task in tasks:
		paths.append(materialize_demo(task, Path(args.data_dir), args.cache_dir, args.token))
	for path in paths:
		print(path)


if __name__ == "__main__":
	main()
