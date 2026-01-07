import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pldm.utils import pick_latest_model


METRICS = ["planning_error_mean", "cross_wall_rate", "init_plan_cross_wall_rate"]


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _output_path_from_config(
    repo_root: Path, config_path: Path, output_root: Optional[str]
):
    cfg = OmegaConf.load(config_path)
    cfg_output_root = output_root or cfg.get("output_root")
    output_dir = cfg.get("output_dir")
    if cfg_output_root is None or output_dir is None:
        raise ValueError("output_root or output_dir missing in config")
    output_root_path = Path(cfg_output_root)
    if not output_root_path.is_absolute():
        output_root_path = repo_root / output_root_path
    return (output_root_path / output_dir).resolve()


def _build_values(args, extra_values=None):
    values = []
    if args.output_root:
        values.append(f"output_root={args.output_root}")
    if args.seed is not None:
        values.append(f"seed={args.seed}")
    if args.epochs is not None:
        values.append(f"epochs={args.epochs}")
    if extra_values:
        values.extend(extra_values)
    return values


def _run_train(repo_root: Path, config_path: Path, values):
    cmd = [sys.executable, "-m", "pldm.train", "--configs", str(config_path)]
    if values:
        cmd += ["--values", *values]
    subprocess.run(cmd, check=True, cwd=repo_root)


def _ensure_wall_trials(repo_root: Path, config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    wall_eval_cfg = cfg.get("eval_cfg", {}).get("wall_planning", None)
    if wall_eval_cfg is None:
        return
    eval_path = wall_eval_cfg.get("set_start_target_path")
    if not eval_path:
        return

    eval_path = _resolve_path(repo_root, eval_path)
    if eval_path.exists():
        return

    n_eval = int(wall_eval_cfg.get("n_envs", 20))
    train_path = repo_root / "PLDM_hieral" / "wall_trials_train.npz"
    seed = cfg.get("seed")
    if seed is None:
        seed = 42
    seed = int(seed)
    cmd = [
        sys.executable,
        str(repo_root / "PLDM_hieral" / "generate_wall_trials.py"),
        "--config",
        str(config_path),
        "--output_train",
        str(train_path),
        "--output_eval",
        str(eval_path),
        "--n_eval",
        str(n_eval),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)


def _load_summary(output_path: Path):
    summary_path = output_path / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    summary_candidates = sorted(output_path.glob("summary_epoch=*.json"))
    if summary_candidates:
        return json.loads(summary_candidates[-1].read_text())
    return {}


def _extract_metrics(summary):
    metrics = {}
    for key, value in summary.items():
        if not isinstance(value, (int, float)):
            continue
        for metric_name in METRICS:
            if metric_name in key:
                metrics[key] = value
    return metrics


def _write_csv(output_root: Path, metrics_l1, metrics_l2):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "tworooms_compare.csv"
    all_keys = sorted(set(metrics_l1) | set(metrics_l2))
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "l1", "l2"])
        for key in all_keys:
            writer.writerow([key, metrics_l1.get(key), metrics_l2.get(key)])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run TwoRooms L1 vs L2 comparison.")
    parser.add_argument(
        "--config_l1",
        default="PLDM_hieral/configs/tworooms_l1.yaml",
        help="Path to L1 config",
    )
    parser.add_argument(
        "--config_l2",
        default="PLDM_hieral/configs/tworooms_l2.yaml",
        help="Path to L2 config",
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mode", choices=["l1", "l2", "both"], default="both")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    config_l1 = _resolve_path(repo_root, args.config_l1)
    config_l2 = _resolve_path(repo_root, args.config_l2)

    output_path_l1 = _output_path_from_config(repo_root, config_l1, args.output_root)
    output_path_l2 = _output_path_from_config(repo_root, config_l2, args.output_root)

    if args.mode in ("l1", "both"):
        _ensure_wall_trials(repo_root, config_l1)
        _run_train(repo_root, config_l1, _build_values(args))

    if args.mode in ("l2", "both"):
        _ensure_wall_trials(repo_root, config_l2)
        latest_ckpt = pick_latest_model(output_path_l1)
        if latest_ckpt is None:
            raise FileNotFoundError(f"No L1 checkpoints found in {output_path_l1}")
        values = _build_values(args, extra_values=[f"load_checkpoint_path={latest_ckpt}"])
        _run_train(repo_root, config_l2, values)

    summary_l1 = _load_summary(output_path_l1)
    summary_l2 = _load_summary(output_path_l2)
    metrics_l1 = _extract_metrics(summary_l1)
    metrics_l2 = _extract_metrics(summary_l2)

    if metrics_l1 or metrics_l2:
        print("Metric,L1,L2")
        for key in sorted(set(metrics_l1) | set(metrics_l2)):
            print(f"{key},{metrics_l1.get(key)},{metrics_l2.get(key)}")
    else:
        print("No planning metrics found in summaries.")

    output_root = (
        output_path_l1.parent if args.output_root is None else Path(args.output_root)
    )
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    csv_path = _write_csv(output_root, metrics_l1, metrics_l2)
    print(f"Saved comparison CSV to {csv_path}")


if __name__ == "__main__":
    main()
