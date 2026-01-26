"""モデル構築のファクトリー関数"""

from typing import Optional

from pldm.models.model_config import ModelConfig
from pldm.models.enums import ModelType


def build_model(
    model_config: ModelConfig,
    input_dim,
    use_propio_pos: bool = False,
    use_propio_vel: bool = False,
    normalizer=None,
):
    """ModelConfigに基づいてモデルを構築

    Args:
        model_config: モデル設定
        input_dim: 入力次元 (C, H, W)
        use_propio_pos: proprioceptive position使用フラグ
        use_propio_vel: proprioceptive velocity使用フラグ
        normalizer: 正規化器

    Returns:
        JEPA | HJEPA | ...: 構築されたモデル
    """
    model_type = model_config.get_model_type_enum()
    config = model_config.get_active_config()

    if model_type == ModelType.JEPA:
        from pldm.models.jepa import JEPA

        return JEPA(
            config=config,
            input_dim=input_dim,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

    elif model_type in (ModelType.HJEPA_V1, ModelType.HJEPA):
        from pldm.models.hjepa import HJEPA

        return HJEPA(
            config=config,
            input_dim=input_dim,
            normalizer=normalizer,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

    # elif model_type == ModelType.HJEPA_V2:
    #     from pldm.models.hjepa_v2 import HJEPA_V2
    #     return HJEPA_V2(...)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
