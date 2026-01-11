# PLDM_hieral

Minimal HJEPA L2 implementation and TwoRooms (wall) comparison workflow.

## フォルダ構成（要点）
- `PLDM_hieral/configs/`: TwoRooms実験設定（L1/L2）
- `PLDM_hieral/run_tworooms_compare.py`: L1→L2の連続実行スクリプト
- `PLDM_hieral/colab_run.ipynb`: Colab実行手順
- `PLDM_hieral/colab_pack.sh`: Colab用のtar作成
- `PLDM_hieral/wall_mediumlast_episode_level1.gif`: L1可視化（最後のエピソード）
- `PLDM_hieral/wall_mediumlast_episode_level2.gif`: L2可視化（最後のエピソード）
- `PLDM_hieral/tworooms_compare.csv`: L1/L2の比較結果

## 結果まとめ（TwoRooms / wall_medium）
`PLDM_hieral/tworooms_compare.csv` の内容:

| metric | L1 | L2 |
| --- | ---: | ---: |
| wall_mediumcross_wall_rate | 0.3199999928474426 | 0.41999998688697815 |
| wall_mediuminit_plan_cross_wall_rate | 0.5299999713897705 | 0.5699999928474426 |
| wall_mediumplanning_error_mean | 527.7476806640625 | 405.44952392578125 |
| wall_mediumplanning_error_mean_rmse | 22.972759246826172 | 20.13577651977539 |

メモ:
- `cross_wall_rate` / `init_plan_cross_wall_rate` は高いほど良い（反対側へ到達できた割合）
- `planning_error_mean` / `rmse` は低いほど良い（ゴール誤差）

## 可視化（GIF）
- L1: `wall_mediumlast_episode_level1.gif`  
  ![L1 last episode](wall_mediumlast_episode_level1.gif)
- L2: `wall_mediumlast_episode_level2.gif`  
  ![L2 last episode](wall_mediumlast_episode_level2.gif)

## Contents
- `PLDM_hieral/configs/tworooms_l1.yaml`: L1-only baseline.
- `PLDM_hieral/configs/tworooms_l2.yaml`: L2 training (L1 frozen).
- `PLDM_hieral/run_tworooms_compare.py`: Run L1, then L2, and print a summary.
- `PLDM_hieral/colab_pack.sh`: Create a tarball for Colab.
- `PLDM_hieral/colab_run.ipynb`: Colab workflow.

## Dataset prep (TwoRooms)
Run from the repo root:

```bash
bash -c "cd pldm_envs/wall && bash presaved_datasets/download_all.sh"
bash -c "cd pldm_envs/wall && bash presaved_datasets/render_all.sh"
```

## Run locally

```bash
python PLDM_hieral/run_tworooms_compare.py --mode both
```

Optional overrides:

```bash
python PLDM_hieral/run_tworooms_compare.py --mode both --epochs 100 --seed 123 --output_root PLDM_hieral/outputs
```

## Colab workflow
1) Create a tarball:

```bash
bash PLDM_hieral/colab_pack.sh
```

2) Upload the tarball to Colab and run `PLDM_hieral/colab_run_(1).ipynb`.

## Outputs
- L1 outputs: `PLDM_hieral/outputs/tworooms_l1`
- L2 outputs: `PLDM_hieral/outputs/tworooms_l2`
- Summaries: `summary.json` and `summary_epoch=*.json` in each output dir
- Comparison CSV: `PLDM_hieral/tworooms_compare.csv`（デフォルト）。`--output_root` を指定した場合はその配下に保存されます。
