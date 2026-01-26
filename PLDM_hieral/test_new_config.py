#!/usr/bin/env python3
"""新しいModelConfig形式の設定ファイルのテスト"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pldm.train import TrainConfig
from pldm.models import ModelConfig, ModelType


def test_config_loading():
    """設定ファイルの読み込みテスト"""

    print("=" * 80)
    print("新しい設定ファイル形式のテスト")
    print("=" * 80)

    # テスト1: HJEPA V1形式（新しいmodel形式）
    print("\n[テスト1] HJEPA V1形式（新しいmodel:形式）")
    print("-" * 80)
    config_path = "PLDM_hieral/configs/tworooms_l1_new_format.yaml"
    try:
        config = TrainConfig.parse_from_file(config_path)
        print(f"✓ 設定ファイル読み込み成功: {config_path}")

        # モデル設定の確認
        assert config.model is not None, "model設定が見つかりません"
        print(f"  - model_type: {config.model.model_type}")

        active_config = config.model.get_active_config()
        print(f"  - active_config type: {type(active_config).__name__}")
        print(f"  - train_l1: {getattr(active_config, 'train_l1', 'N/A')}")
        print(f"  - disable_l2: {getattr(active_config, 'disable_l2', 'N/A')}")

        model_type_enum = config.model.get_model_type_enum()
        print(f"  - model_type_enum: {model_type_enum}")

        print("✓ HJEPA V1形式のテスト成功")
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テスト2: JEPA形式（新しいmodel形式）
    print("\n[テスト2] JEPA形式（新しいmodel:形式）")
    print("-" * 80)
    config_path = "PLDM_hieral/configs/tworooms_jepa_new_format.yaml"
    try:
        config = TrainConfig.parse_from_file(config_path)
        print(f"✓ 設定ファイル読み込み成功: {config_path}")

        # モデル設定の確認
        assert config.model is not None, "model設定が見つかりません"
        print(f"  - model_type: {config.model.model_type}")

        active_config = config.model.get_active_config()
        print(f"  - active_config type: {type(active_config).__name__}")
        print(f"  - action_dim: {active_config.action_dim}")
        print(f"  - momentum: {active_config.momentum}")

        model_type_enum = config.model.get_model_type_enum()
        print(f"  - model_type_enum: {model_type_enum}")
        assert model_type_enum == ModelType.JEPA, "model_typeがJEPAではありません"

        print("✓ JEPA形式のテスト成功")
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テスト3: 旧形式（後方互換性）
    print("\n[テスト3] 旧形式（後方互換性: hjepa:形式）")
    print("-" * 80)
    config_path = "PLDM_hieral/configs/tworooms_l1.yaml"
    try:
        config = TrainConfig.parse_from_file(config_path)
        print(f"✓ 設定ファイル読み込み成功: {config_path}")

        # 旧形式でも自動変換されているか確認
        assert config.model is not None, "model設定が見つかりません（自動変換失敗）"
        print(f"  - model_type: {config.model.model_type}")
        print(f"  - 自動変換: hjepa: → model: ✓")

        active_config = config.model.get_active_config()
        print(f"  - active_config type: {type(active_config).__name__}")

        print("✓ 旧形式の後方互換性テスト成功")
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ すべてのテストが成功しました！")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
