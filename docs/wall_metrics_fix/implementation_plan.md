# 壁越え指標（Wall Crossing Metrics）の修正計画

## ゴール
ユーザーが `pldm/planning/wall/mpc.py` に追加した壁越え指標（`efficiency_score`, `first_crossing_steps`）の実装を確認し、以下の問題を修正する。
1. **ロジックの誤り**: `first_crossing_steps` の計算におけるOff-by-oneエラー（1ステップ多くカウントされている）。
2. **出力の欠損**: `PLDM_hieral/run_tworooms_compare_feedback.py` の集計対象メトリクスに含まれていないため、出力されない。また、`success_rate` も同様に除外されていることが判明。

## ユーザーレビューが必要な事項
> [!IMPORTANT]
> `first_crossing_steps` の計算式を `t + 1` から `t` に変更します。これにより、以前の値よりも1小さくなります（例: 最初のステップで越えた場合、値は1になります）。

## 変更内容
### pldm
#### [MODIFY] [mpc.py](file:///home/owner/devws/cuda/WM-JEPA/pldm/planning/wall/mpc.py)
- `_construct_report` メソッド内の `first_crossing_steps` 計算において、`t + 1` を `t` に修正します。
- 理由: `locations[0]` が初期状態（ステップ0）であるため、インデックス `t` が実際の経過ステップ数と一致するため。

### PLDM_hieral
#### [MODIFY] [run_tworooms_compare_feedback.py](file:///home/owner/devws/cuda/WM-JEPA/PLDM_hieral/run_tworooms_compare_feedback.py)
- `METRICS` リストに `efficiency_score`、`first_crossing_steps`、**`success_rate`** を追加します。

## 検証計画
### 自動テスト
- ユーザーが提示したコマンド（エポック数を減らして）を実行し、出力に指標が含まれるか確認します。
  ```bash
  python PLDM_hieral/run_tworooms_compare_feedback.py \
      --config_l1 PLDM_hieral/configs/tworooms_l1.yaml \
      --config_l2 PLDM_hieral/configs/tworooms_feedback.yaml \
      --output_root ./PLDM_hieral/output_feedback_verify \
      --l2_from_scratch \
      --mode both \
      --epochs 5
  ```
- CSVファイルおよびコンソール出力に `efficiency_score`、`first_crossing_steps`、`success_rate` が存在することを確認します。

### 手動検証
- 出力された `first_crossing_steps` が妥当な値（>= 1）であることを確認します。
