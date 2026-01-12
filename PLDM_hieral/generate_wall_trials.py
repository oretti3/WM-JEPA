import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pldm_envs.wall.wall import DotWall


def _load_wall_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    wall_cfg = cfg["data"]["wall_config"]
    eval_cfg = cfg["eval_cfg"]["wall_planning"]
    levels = eval_cfg.get("levels", "medium")
    level = levels.split(",")[0] if isinstance(levels, str) else "medium"
    return wall_cfg, level


def _build_env(wall_cfg, level: str):
    env = DotWall(
        rng=None,
        border_wall_loc=wall_cfg["border_wall_loc"],
        wall_width=wall_cfg["wall_width"],
        door_space=wall_cfg["door_space"],
        wall_padding=wall_cfg["wall_padding"],
        img_size=wall_cfg["img_size"],
        fix_wall=wall_cfg["fix_wall"],
        cross_wall=True,
        level=level,
        n_steps=wall_cfg["n_steps"],
        action_step_mean=wall_cfg["action_step_mean"],
        max_step_norm=wall_cfg["action_upper_bd"],
        fix_wall_location=wall_cfg["fix_wall_location"],
        fix_door_location=wall_cfg["fix_door_location"],
    )
    return env


def _generate_trials(env: DotWall, n_trials: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trials = np.zeros((n_trials, 6), dtype=np.float32)
    for i in range(n_trials):
        env.reset()
        start = env.dot_position.detach().cpu().numpy()
        target = env.target_position.detach().cpu().numpy()
        wall_x = float(env.wall_x.detach().cpu().item())
        door_y = float(env.hole_y.detach().cpu().item())
        trials[i] = np.array(
            [start[0], start[1], target[0], target[1], wall_x, door_y],
            dtype=np.float32,
        )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TwoRooms trials.")
    parser.add_argument(
        "--config",
        default="PLDM_hieral/configs/tworooms_l1.yaml",
        help="Config path for wall parameters.",
    )
    parser.add_argument("--n-train", "--n_train", dest="n_train", type=int, default=None)
    parser.add_argument("--n-eval", "--n_eval", dest="n_eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-train",
        "--output_train",
        dest="output_train",
        default="PLDM_hieral/wall_trials_train.npz",
    )
    parser.add_argument(
        "--output-eval",
        "--output_eval",
        dest="output_eval",
        default="PLDM_hieral/wall_trials_eval.npz",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    wall_cfg, level = _load_wall_config(config_path)
    if args.n_train is None:
        args.n_train = int(wall_cfg.get("size", 20000))

    env = _build_env(wall_cfg, level=level)

    train_trials = _generate_trials(env, args.n_train, seed=args.seed)
    eval_trials = _generate_trials(env, args.n_eval, seed=args.seed + 1)

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_eval).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_train, trials=train_trials, seed=args.seed, level=level)
    np.savez(args.output_eval, trials=eval_trials, seed=args.seed + 1, level=level)

    print(f"Saved train trials to {args.output_train}")
    print(f"Saved eval trials to {args.output_eval}")


if __name__ == "__main__":
    main()
