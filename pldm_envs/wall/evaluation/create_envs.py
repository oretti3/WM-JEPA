import numpy as np
import torch
from pldm_envs.wall.wall import DotWall
from pldm_envs.wall.wrappers import NormEvalWrapper


def construct_eval_envs(
    seed,
    wall_config,
    n_envs: int,
    level: str,
    cross_wall: bool = True,
    start_ends=None,
    normalizer=None,
):
    if start_ends is None:
        rng = np.random.default_rng(seed)
        envs = [
            DotWall(
                rng=rng,
                border_wall_loc=wall_config.border_wall_loc,
                wall_width=wall_config.wall_width,
                door_space=wall_config.door_space,
                wall_padding=wall_config.wall_padding,
                img_size=wall_config.img_size,
                fix_wall=wall_config.fix_wall,
                cross_wall=cross_wall,
                level=level,
                n_steps=wall_config.n_steps,
                action_step_mean=wall_config.action_step_mean,
                max_step_norm=wall_config.action_upper_bd,
                fix_wall_location=wall_config.fix_wall_location,
                fix_door_location=wall_config.fix_door_location,
            )
            for _ in range(n_envs)
        ]

        [e.reset() for e in envs]

        if normalizer is not None:
            envs = [NormEvalWrapper(e, normalizer=normalizer) for e in envs]

        return envs
    else:
        start_ends = np.asarray(start_ends)
        if start_ends.shape[0] < n_envs:
            raise ValueError(
                f"start_ends has {start_ends.shape[0]} rows, needs {n_envs}"
            )
        if start_ends.shape[1] < 4:
            raise ValueError("start_ends must have at least 4 columns")

        envs = []
        for i in range(n_envs):
            start_x, start_y, target_x, target_y = start_ends[i][:4]
            wall_x = (
                start_ends[i][4]
                if start_ends.shape[1] >= 5
                else wall_config.fix_wall_location
            )
            door_y = (
                start_ends[i][5]
                if start_ends.shape[1] >= 6
                else wall_config.fix_door_location
            )

            env = DotWall(
                rng=None,
                border_wall_loc=wall_config.border_wall_loc,
                wall_width=wall_config.wall_width,
                door_space=wall_config.door_space,
                wall_padding=wall_config.wall_padding,
                img_size=wall_config.img_size,
                fix_wall=wall_config.fix_wall,
                cross_wall=cross_wall,
                level=level,
                n_steps=wall_config.n_steps,
                action_step_mean=wall_config.action_step_mean,
                max_step_norm=wall_config.action_upper_bd,
                fix_wall_location=int(wall_x),
                fix_door_location=int(door_y),
            )

            env.reset()
            env.wall_x = torch.tensor(wall_x, device=env.device)
            env.hole_y = torch.tensor(door_y, device=env.device)
            env.left_wall_x = env.wall_x - env.wall_width // 2
            env.right_wall_x = env.wall_x + env.wall_width // 2
            env.wall_img = env._render_walls(env.wall_x, env.hole_y)
            env.dot_position = torch.tensor(
                [start_x, start_y], dtype=torch.float32, device=env.device
            )
            env.target_position = torch.tensor(
                [target_x, target_y], dtype=torch.float32, device=env.device
            )
            env.position_history = [env.dot_position]
            envs.append(env)

        if normalizer is not None:
            envs = [NormEvalWrapper(e, normalizer=normalizer) for e in envs]

        return envs
