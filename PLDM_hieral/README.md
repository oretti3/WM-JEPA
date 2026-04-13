# PLDM_hieral

H-JEPA (Hierarchical JEPA) の実験用スクリプト・設定ファイル群です。
TwoRooms環境 (壁で仕切られた2部屋ナビゲーション) において、ベースライン (L1) と階層化モデル (Hieral) の性能比較を行います。

## 実験スクリプト

| スクリプト | 用途 |
|:---|:---|
| `run_tworooms_compare.py` | L1 vs Hieral (サブゴール方式) の比較実験 |
| `run_tworooms_compare_feedback.py` | L1 vs Hieral (フィードバック方式) の比較実験 |
| `generate_wall_trials.py` | 評価用の固定スタート・ゴールエピソード生成 |
| `verify_matched.py` | パラメータ数の検証 |

## 設定ファイル (configs/)

| ファイル | 説明 |
|:---|:---|
| `tworooms_l1.yaml` | L1ベースライン |
| `tworooms_l1_2m.yaml` | L1 (2Mパラメータ) |
| `tworooms_l1_6m.yaml` | L1 (6Mパラメータ) |
| `tworooms_l2.yaml` | L2 階層化 (サブゴール方式) |
| `tworooms_l2_2m.yaml` | L2 (2Mパラメータ) |
| `tworooms_feedback.yaml` | L2 階層化 (フィードバック方式) |
| `tworooms_feedback_matched.yaml` | フィードバック (パラメータ数マッチ, 2.2M) |
| `tworooms_feedback_matched_v2.yaml` | フィードバック (パラメータ数マッチ v2) |

## 実行例

```bash
# サブゴール方式
python PLDM_hieral/run_tworooms_compare.py --mode both

# フィードバック方式
python PLDM_hieral/run_tworooms_compare_feedback.py \
  --config_l1 PLDM_hieral/configs/tworooms_l1.yaml \
  --config_l2 PLDM_hieral/configs/tworooms_feedback.yaml \
  --l2_from_scratch \
  --mode both \
  --epochs 6 \
  --output_root ./PLDM_hieral/output_feedback_ep6
```

## Colab

```bash
bash PLDM_hieral/colab_pack.sh            # サブゴール版
bash PLDM_hieral/colab_pack_feedback.sh    # フィードバック版
```

ノートブック: `colab_run_(1).ipynb`, `colab_run_feedback.ipynb`

## 可視化 (GIF)

- L1: `wall_mediumlast_episode_level1.gif`
- Hieral: `wall_mediumlast_episode_level2.gif`
