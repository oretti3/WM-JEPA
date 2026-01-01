# PLDM_hieral

Minimal HJEPA L2 implementation and TwoRooms (wall) comparison workflow.

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

2) Upload the tarball to Colab and run `PLDM_hieral/colab_run.ipynb`.

## Outputs
- L1 outputs: `PLDM_hieral/outputs/tworooms_l1`
- L2 outputs: `PLDM_hieral/outputs/tworooms_l2`
- Summaries: `summary.json` and `summary_epoch=*.json` in each output dir
- Comparison CSV: `PLDM_hieral/outputs/tworooms_compare.csv` (or `--output_root`)
