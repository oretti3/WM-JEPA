# Experiment Configuration Walkthrough

実験計画に基づき、以下の3つの比較グループに対応するConfigファイルを生成しました。
全ファイルは `PLDM_hieral/configs/` に保存されています。

## 1. Horizon Comparison
MPPIの計画期間（Horizon）の影響を比較します。
**Base Setting**: 6M Model, 1500k Data, 1 Epoch

| Config File | Horizon |
| :--- | :--- |
| `tworooms_l{1,2}_6m_h5.yaml` | 5 |
| `tworooms_l{1,2}_6m_h24.yaml` | 24 |
| `tworooms_l{1,2}_6m_h48.yaml` | 48 |
| `tworooms_l{1,2}_6m_h96.yaml` | 96 |
| `tworooms_l{1,2}_6m_h192.yaml` | 192 |

## 2. Data Scale Comparison
データ量の影響を比較します。1500k/1epの学習量に合わせるためEpoch数を調整しています。
**Base Setting**: 6M Model, Horizon 96

| Config File | Dataset | Epochs | Note |
| :--- | :--- | :--- | :--- |
| `tworooms_l{1,2}_6m_d634.yaml` | 634 | 2366 | 既存データセット相当 |
| `tworooms_l{1,2}_6m_d20312.yaml` | 20312 | 74 | ※データファイル要準備 |
| `tworooms_l{1,2}_6m_d1500k.yaml` | 1500k | 1 | Baseline |

> [!WARNING]
> `ds_size_20312.npz` および `ds_size_1500K.npz` は現在存在しない可能性があります。実験実行前に当該パスにデータセットを配置してください。

## 3. Model Size Comparison
モデルサイズ（パラメータ数）の影響を比較します。
**Base Setting**: Horizon 96, 1500k Data, 1 Epoch

| Config File | Model Size |
| :--- | :--- |
| `tworooms_l{1,2}_2m_h96.yaml` | 2M (Small) |
| `tworooms_l{1,2}_6m_h96.yaml` | 6M (Base) |

## Automation Script
Configファイルの生成には以下のスクリプトを使用しました。条件変更時の再生成等にご利用ください。
- [generate_configs.py](file:///home/owner/devws/WM-JEPA/PLDM_hieral/configs/generate_configs.py)
