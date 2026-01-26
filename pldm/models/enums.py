from pldm.configs import ConfigBase
from dataclasses import dataclass
from typing import Optional, NamedTuple
from enum import Enum
import torch


@dataclass
class PredictorConfig(ConfigBase):
    predictor_arch: str = "rnnV3"
    predictor_subclass: str = "a"
    predictor_ln: bool = False
    rnn_state_dim: int = 512
    rnn_converter_arch: str = ""
    z_discrete: bool = False
    z_discrete_dim: int = 16
    z_discrete_dists: int = 16
    z_dim: int = 0
    z_min_std: float = 0.1
    posterior_drop_p: float = 0.0
    prior_arch: str = "512"
    posterior_arch: str = "512"
    posterior_input_type: str = "term_states"
    posterior_input_dim: Optional[int] = None
    action_encoder_arch: str = ""
    residual: bool = False
    rnn_layers: int = 1
    tie_backbone_ln: bool = False


class PredictorOutput(NamedTuple):
    predictions: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    propio_component: Optional[torch.Tensor] = None
    prior_mus: Optional[torch.Tensor] = None
    prior_vars: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    posterior_mus: Optional[torch.Tensor] = None
    posterior_vars: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None
    posteriors: Optional[torch.Tensor] = None


class ModelType(str, Enum):
    """モデルアーキテクチャのタイプ"""

    JEPA = "jepa"
    HJEPA_V1 = "hjepa_v1"
    HJEPA = "hjepa"  # エイリアス（hjepa_v1と同じ）
    # HJEPA_V2 = "hjepa_v2"  # 将来追加
