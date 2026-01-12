import argparse
import dataclasses
from typing import Tuple

import torch

from pldm.models.jepa import JEPA, JEPAConfig

DEFAULT_OUTPUT_DIM = 1024
DEFAULT_INPUT_DIM: Tuple[int, int, int] = (2, 65, 65)


def build_jepa_6m_config(
    base: JEPAConfig | None = None,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> JEPAConfig:
    cfg = dataclasses.replace(base) if base is not None else JEPAConfig()
    cfg.backbone = dataclasses.replace(cfg.backbone)
    cfg.predictor = dataclasses.replace(cfg.predictor)

    cfg.backbone.arch = "impala"
    cfg.backbone.fc_output_dim = output_dim
    cfg.backbone.final_ln = True

    cfg.predictor.predictor_arch = "rnnV2"
    cfg.predictor.predictor_ln = True
    cfg.predictor.tie_backbone_ln = True

    cfg.action_dim = 2
    return cfg


def build_jepa_6m(
    input_dim: Tuple[int, int, int] = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> JEPA:
    cfg = build_jepa_6m_config(output_dim=output_dim)
    return JEPA(cfg, input_dim=input_dim)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JEPA ~6M baseline.")
    parser.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    args = parser.parse_args()

    model = build_jepa_6m(output_dim=args.output_dim)
    total, trainable = count_params(model)
    print(f"JEPA output_dim={args.output_dim} total_params={total:,} trainable={trainable:,}")


if __name__ == "__main__":
    main()
