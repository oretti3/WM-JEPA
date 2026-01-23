# Implementation Plan - Experiment Configuration

実験計画に基づき、複数の設定ファイル (config) を作成します。
プロジェクト構造と既存の config ファイルの解析は完了しています。

## User Review Required

**[UPDATE]** ユーザー要望に基づき、**「1500Kデータセット 1 Epoch」を基本設定**として実験計画を再構築しました。
データ量比較実験（Group 2）については、総学習ステップ数をこの基準（1500Kステップ）に合わせる形でEpoch数を調整します。(`train.py`の修正は行わず、Epoch数設定で対応します)

### Experiment Matrix

**Base Setting (基準)**:
- Dataset: 1500K (Size $\approx 1,500,000$)
- Epochs: 1
- Est. Total Steps: $\approx 1,500,000 / BatchSize$

#### Group 1: Horizon Comparison (Model: 6M, Data: 1500K)
計画期間の長さによる影響比較。**基準の1500Kデータを使用。**
| Config Name | Model | Horizon | Data | Epochs |
| :--- | :--- | :--- | :--- | :--- |
| `tworooms_l{1,2}_6m_h5.yaml` | 6M | **5** (Very Short) | 1500K | 1 |
| `tworooms_l{1,2}_6m_h24.yaml` | 6M | **24** (Short) | 1500K | 1 |
| `tworooms_l{1,2}_6m_h48.yaml` | 6M | **48** (Medium) | 1500K | 1 |
| `tworooms_l{1,2}_6m_h96.yaml` | 6M | **96** (Base) | 1500K | 1 |
| `tworooms_l{1,2}_6m_h192.yaml` | 6M | **192** (Long) | 1500K | 1 |

#### Group 2: Data Scale Comparison (Model: 6M, Horizon: 96)
データ量によるスケーリング比較。**総ステップ数を1500Kサンプル分に統一。**
| Config Name | Model | Horizon | Data | Epochs | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tworooms_l{1,2}_6m_d634.yaml` | 6M | 96 | **634** | **2366** | $1.5M / 634 \approx 2366$ |
| `tworooms_l{1,2}_6m_d20312.yaml` | 6M | 96 | **20312** | **74** | $1.5M / 20312 \approx 74$ |
| `tworooms_l{1,2}_6m_d1500k.yaml` | 6M | 96 | **1500K** | **1** | Baseline |

#### Group 3: Model Size Comparison (Horizon: 96, Data: 1500K)
モデルサイズ(2M/6M)の比較。**基準の1500Kデータを使用。**
| Config Name | Model | Horizon | Data | Epochs |
| :--- | :--- | :--- | :--- | :--- |
| `tworooms_l{1,2}_2m_h96.yaml` | **2M** | 96 | 1500K | 1 |
| `tworooms_l{1,2}_6m_h96.yaml` | **6M** | 96 | 1500K | 1 |

**Parameter Definitions**:
- **Dataset Paths**:
    - 634: `.../ds_size_634.npz`
    - 20312: `.../ds_size_20312.npz`
    - 1500K: `.../ds_size_1500K.npz`
- **2M**: `backbone_width_factor: 1`, `predictor_subclass: "128-128"`, `l2_arch: "128-128"`
- **6M**: `backbone_width_factor: 2`, `predictor_subclass: "512-512"`, `l2_arch: "512-512"`

---


## Proposed Changes

## Proposed Changes

### Configuration Files
#### [MODIFY] [tworooms configurations](file:///home/owner/devws/WM-JEPA/PLDM_hieral/configs/)
- Create new configuration files based on the experiment matrix above.
- Base files: `tworooms_l1.yaml`, `tworooms_l2.yaml` (or existing `_6m` variants)

### Training Logic
- (`train.py` modification removed as we use epoch adjustment logic)

ユーザーからの入力に基づき、以下のディレクトリに新しい config ファイルを作成します。

### Config Files
#### [NEW] [config_name].yaml
- ベース config をコピーし、指定されたパラメータを変更します。

## Verification Plan

### Automated Tests
- `verify_configs.py` (またはこれを修正したもの) を実行し、作成した config が正しく読み込め、モデルが期待通りの構成（パラメータ数など）になっているかを確認します。
