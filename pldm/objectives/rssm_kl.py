from dataclasses import dataclass
from typing import NamedTuple, List

import torch

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult


class RSSMKLLossInfo(NamedTuple):
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss_name: str = "rssm_kl"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_kl_loss": self.kl_loss.item(),
        }


@dataclass
class RSSMKLObjectiveConfig(ConfigBase):
    coeff: float = 1.0
    free_nats: float = 0.0


class RSSMKLObjective(torch.nn.Module):
    def __init__(self, config: RSSMKLObjectiveConfig, name_prefix: str = ""):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix

    def __call__(self, _batch, result: List[ForwardResult]) -> RSSMKLLossInfo:
        result = result[-1]
        pred_output = result.pred_output
        if (
            pred_output is None
            or pred_output.prior_mus is None
            or pred_output.prior_vars is None
            or pred_output.posterior_mus is None
            or pred_output.posterior_vars is None
        ):
            device = (
                pred_output.prior_mus.device
                if pred_output is not None and pred_output.prior_mus is not None
                else torch.device("cpu")
            )
            zero = torch.zeros(1, device=device)
            return RSSMKLLossInfo(
                total_loss=zero, kl_loss=zero, name_prefix=self.name_prefix
            )

        prior = torch.distributions.Normal(
            pred_output.prior_mus, pred_output.prior_vars
        )
        posterior = torch.distributions.Normal(
            pred_output.posterior_mus, pred_output.posterior_vars
        )
        kl = torch.distributions.kl_divergence(posterior, prior)
        kl = kl.mean(dim=-1).mean()
        if self.config.free_nats > 0:
            kl = torch.clamp(kl, min=self.config.free_nats)
        total_loss = self.config.coeff * kl

        return RSSMKLLossInfo(
            total_loss=total_loss,
            kl_loss=kl,
            loss_name="rssm_kl",
            name_prefix=self.name_prefix,
        )
