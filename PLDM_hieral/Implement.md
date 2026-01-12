# HJEPA Level2 + TwoRooms Comparison (Implementation Guide)

## Goals
- Implement the missing HJEPA level2 path (model + training) in a minimal, non-breaking way.
- Run TwoRooms (wall) experiments and compare L1-only vs L1+L2.
- Provide runnable scripts and Colab artifacts under PLDM_hieral.

## Assumptions
- "HLEPA" in the request means "HJEPA".
- TwoRooms corresponds to the wall environment (pldm/configs/wall).
- Default behavior must remain unchanged when hjepa.disable_l2=true.

## Current gaps in the repo
- pldm/models/hjepa.py only builds level1; forward_prior/forward_posterior skip L2.
- RSSMPredictor references predictor_l2/posterior_l2/decoder that do not exist.
- Trainer only builds objectives_l1 and never computes L2 losses.
- No L2 config or run scripts are provided.

## Proposed design (minimal + faithful)
1) L2 uses an RSSM-style predictor over downsampled L1 encodings.
   - Use step_skip to build the L2 time grid.
   - Flatten L1 encodings for L2 rnn_state.
   - predictor_l2 = RSSMPredictor(...)
   - posterior_l2 and decoder are small MLPs (build_mlp).
2) L2 outputs are wrapped as JEPA ForwardResult so existing objectives can be reused.
3) L1 training remains unchanged. L2 is opt-in with new objectives and can freeze L1.

## Implementation steps

### A) Model: HJEPA level2
Files: pldm/models/hjepa.py (primary), pldm/models/predictors.py (optional helper)

- Extend HJEPAConfig with L2 params:
  - l2_z_dim (int)
  - l2_rnn_state_dim (int, default to flattened L1 repr dim)
  - l2_min_var (float)
  - l2_use_action_only (bool)
  - l2_posterior_arch (str like "512-512")
  - l2_decoder_arch (str like "512-512")
  - l2_use_actions (bool)
  - l2_action_agg ("concat" for now)
- Extend ForwardResult in hjepa.py to include level2 (Optional).
- In HJEPA.__init__ (when not disable_l2):
  - Compute l1_repr_dim using flatten_conv_output on level1 output (store as self.l2_repr_dim).
  - predictor_l2 = RSSMPredictor(
      rnn_state_dim=self.l2_repr_dim,
      z_dim=l2_z_dim,
      action_dim=level1.action_dim * step_skip (or 0 if not using actions),
      min_var=l2_min_var,
      use_action_only=l2_use_action_only,
    )
  - posterior_l2: MLP that maps concat(enc_next, rnn_state) -> [mu, raw_var].
  - decoder: MLP that maps concat(rnn_state, z) -> predicted L1 encoding.
- In HJEPA.forward_posterior:
  - If L2 enabled, always compute full L1 encodings with encode_only=True.
  - Downsample encs every step_skip: encs_l2 = encs[::step_skip].
  - Aggregate actions into L2 actions (concat step_skip actions).
  - Roll L2 posterior:
    - rnn_state = encs_l2[0]
    - for i in range(T2 - 1):
      - rnn_state, prior_mu, prior_var = predictor_l2(sampled_z, actions_l2[i], rnn_state)
      - post_mu, post_var = posterior_l2(encs_l2[i+1], rnn_state)
      - z = Normal(post_mu, softplus(post_var) + l2_min_var).sample()
      - pred = decoder(rnn_state, z)
  - Stack preds to (T2, B, l1_repr_dim) and wrap:
    - BackboneOutput(encodings=encs_l2)
    - PredictorOutput(predictions=preds)
    - JEPA ForwardResult for level2
- In HJEPA.forward_prior (optional but recommended):
  - Support L2 prior rollout with actions_l2 or latents.

### B) Training: L2 objectives
Files: pldm/train.py, pldm/objectives/__init__.py (if adding config field)

- Add objectives_l2: ObjectivesConfig to TrainConfig (default empty).
- Build objectives_l2 in Trainer.__init__ when L2 is enabled:
  - repr_dim = model.l2_repr_dim
- In training loop:
  - If not disable_l2, compute L2 loss_infos:
    - [obj(batch, [forward_result.level2]) for obj in objectives_l2]
  - Add these losses to total_loss.
- Keep L1-only path unchanged; L2 should be opt-in.

### C) Configs for TwoRooms
Place new configs under PLDM_hieral/configs to avoid touching originals.

- Baseline (L1-only): use an existing wall config as-is (e.g., pldm/configs/wall/icml/ds_634.yaml).
- L2 config:
  - hjepa.disable_l2: false
  - hjepa.train_l1: false
  - hjepa.freeze_l1: true
  - objectives_l2: include Prediction (and optionally VICReg)
  - load_checkpoint_path: path to baseline checkpoint
  - set l2_* params (z_dim, decoder/posterior arch)
- Keep dataset and eval_cfg the same between L1 and L2.

### D) Run script (PLDM_hieral)
Create `PLDM_hieral/run_tworooms_compare.py`.

- CLI args:
  - --config_l1, --config_l2
  - --output_root, --seed, --epochs (optional overrides)
  - --mode {l1,l2,both}
- Steps:
  - Run L1 training via: python pldm/train.py --configs <config_l1> --values ...
  - Locate latest L1 checkpoint (pldm/utils.pick_latest_model).
  - Run L2 training with load_checkpoint_path set to the L1 checkpoint.
  - Read summary JSON and print key metrics (error_mean, cross_wall_rate, init_plan_cross_wall_rate).

### E) Colab support
Create two artifacts in PLDM_hieral:

1) `colab_pack.sh`
- Create a tarball with PLDM + PLDM_hieral + requirements.
- Print instructions for manual upload (keeps repo intact).

2) `colab_run.ipynb`
- Upload tarball (or git clone as an alternative).
- Install deps:
  - pip install -r requirements.txt
  - pip install gdown
- Download TwoRooms dataset:
  - bash pldm_envs/wall/presaved_datasets/download_all.sh
  - bash pldm_envs/wall/presaved_datasets/render_all.sh
- Run run_tworooms_compare.py.
- (Optional) mount Drive and copy outputs.

### F) README
Create `PLDM_hieral/README.md` with:
- Purpose and scope
- Script overview and usage
- Dataset prep for TwoRooms
- Colab workflow (pack + notebook)
- Expected outputs and where to find summary JSON

## Comparison protocol
- Train baseline L1-only (disable_l2 true).
- Train L2 with L1 frozen and identical data/epochs.
- Compare metrics:
  - planning: error_mean, cross_wall_rate, init_plan_cross_wall_rate
  - probing losses (from summary JSON)
- Print a short table and optionally save CSV in PLDM_hieral/outputs.

## Acceptance criteria
- L1-only training still works with original configs.
- L2 training runs end-to-end in TwoRooms without runtime errors.
- run_tworooms_compare.py prints a comparison summary.
- Colab notebook runs the pipeline with minimal manual steps.
