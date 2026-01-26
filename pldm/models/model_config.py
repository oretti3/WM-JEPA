"""複数のモデルアーキテクチャを統一的に扱うための設定"""

from dataclasses import dataclass, field
from typing import Optional

from pldm.configs import ConfigBase
from pldm.models.enums import ModelType
from pldm.models.jepa import JEPAConfig
from pldm.models.hjepa import HJEPAConfig


@dataclass
class ModelConfig(ConfigBase):
    """モデルアーキテクチャの切り替えを管理する設定

    使用例:
        # JEPA使用
        model = ModelConfig(model_type="jepa", jepa=JEPAConfig(...))

        # HJEPA V1使用
        model = ModelConfig(model_type="hjepa_v1", hjepa_v1=HJEPAConfig(...))
    """

    model_type: str = "hjepa_v1"  # デフォルトは既存実装

    # 各モデル用の設定（使用されるものだけ有効化）
    jepa: Optional[JEPAConfig] = None
    hjepa_v1: Optional[HJEPAConfig] = None
    # hjepa_v2: Optional[HJEPA_V2Config] = None  # 将来追加

    def __post_init__(self):
        """使用するモデルのconfigを初期化（未指定時）"""
        # model_typeの正規化（"hjepa" → "hjepa_v1"）
        if self.model_type == "hjepa":
            self.model_type = "hjepa_v1"

        model_type = ModelType(self.model_type)

        # 使用するconfigが未設定の場合はデフォルト生成
        if model_type == ModelType.JEPA:
            if self.jepa is None:
                self.jepa = JEPAConfig()
        elif model_type in (ModelType.HJEPA_V1, ModelType.HJEPA):
            if self.hjepa_v1 is None:
                self.hjepa_v1 = HJEPAConfig()
        # elif model_type == ModelType.HJEPA_V2:
        #     if self.hjepa_v2 is None:
        #         raise ValueError("hjepa_v2 requires explicit config")

    def get_active_config(self):
        """使用中のモデル設定を取得

        Returns:
            JEPAConfig | HJEPAConfig | ...: アクティブなモデルの設定

        Raises:
            ValueError: model_typeが不正、または対応するconfigがNone
        """
        model_type = ModelType(self.model_type)

        if model_type == ModelType.JEPA:
            if self.jepa is None:
                raise ValueError("jepa config is None but model_type='jepa'")
            return self.jepa

        elif model_type in (ModelType.HJEPA_V1, ModelType.HJEPA):
            if self.hjepa_v1 is None:
                raise ValueError("hjepa_v1 config is None but model_type='hjepa_v1'")
            return self.hjepa_v1

        # elif model_type == ModelType.HJEPA_V2:
        #     if self.hjepa_v2 is None:
        #         raise ValueError("hjepa_v2 config is None")
        #     return self.hjepa_v2

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def get_model_type_enum(self) -> ModelType:
        """ModelType enumを取得"""
        return ModelType(self.model_type)
