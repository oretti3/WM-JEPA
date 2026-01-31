# 壁越え指標の修正と有効化

- [/] 調査と原因特定 <!-- id: 0 -->
    - [x] `pldm/planning/wall/mpc.py` の実装確認 (Off-by-oneエラーの特定) <!-- id: 1 -->
    - [x] output欠損の原因特定 (`METRICS`リストへの未追加) <!-- id: 2 -->
- [x] 修正の実施 <!-- id: 3 -->
    - [x] `pldm/planning/wall/mpc.py` の `first_crossing_steps` 計算修正 <!-- id: 4 -->
    - [x] `PLDM_hieral/run_tworooms_compare_feedback.py` の `METRICS` リスト更新 <!-- id: 5 -->
- [ ] 検証 <!-- id: 6 -->
    - [ ] 学習スクリプトの実行と指標出力の確認 <!-- id: 7 -->
