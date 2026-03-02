from typing import Optional, NamedTuple
import dataclasses

import torch

from pldm.models.encoders.encoders import build_backbone
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.enums import PredictorConfig, PredictorOutput
from pldm.models.jepa import JEPA, ForwardResult as JEPAForwardResult
from pldm.models.predictors import build_predictor
from pldm.models.utils import flatten_conv_output
from pldm.models.hjepa import HJEPAConfig


class ForwardResult(NamedTuple):
    level1: Optional[JEPAForwardResult]
    level2: Optional[JEPAForwardResult] = None


class _HierarchicalPredictorProxy:
    """
    Expose a JEPA-like predictor API while delegating rollouts to HJEPAFeedback.
    Used by MPPI/SGD planners that call model.predictor.forward_multiple.
    """

    def __init__(self, hjepa: "HJEPAFeedback"):
        self._hjepa = hjepa
        self._predictor = hjepa.level1.predictor
        self.action_dim = self._predictor.action_dim

    @property
    def training(self):
        return self._predictor.training

    def train(self, mode: bool = True):
        self._predictor.train(mode)
        if self._hjepa.l2_predictor is not None:
            self._hjepa.l2_predictor.train(mode)
        return self

    def forward_multiple(
        self,
        state_encs: torch.Tensor,
        actions: Optional[torch.Tensor],
        T: int,
        latents: Optional[torch.Tensor] = None,
        flatten_output: bool = False,
        compute_posterior: bool = False,
    ):
        if compute_posterior:
            raise NotImplementedError(
                "Hierarchical predictor proxy only supports prior rollout"
            )

        if state_encs.dim() >= 3:
            has_time_dim = (
                state_encs.shape[0] == 1
                or (
                    actions is not None
                    and state_encs.shape[0]
                    in (actions.shape[0], actions.shape[0] + 1)
                )
            )
            input_states = state_encs[0] if has_time_dim else state_encs
        elif state_encs.dim() == 2:
            input_states = state_encs
        else:
            raise ValueError(f"Unexpected state_encs shape: {tuple(state_encs.shape)}")

        result = self._hjepa.forward_prior(
            input_states=input_states,
            actions=actions,
            T=T,
            repr_input=True,
            latents=latents,
            level="l1",
        )
        pred_output = result.level1.pred_output
        if pred_output is None:
            raise RuntimeError("Hierarchical forward_prior returned no pred_output")

        if not flatten_output:
            return pred_output

        predictions = flatten_conv_output(pred_output.predictions)
        obs_component = (
            None
            if pred_output.obs_component is None
            else flatten_conv_output(pred_output.obs_component)
        )
        propio_component = (
            None
            if pred_output.propio_component is None
            else flatten_conv_output(pred_output.propio_component)
        )
        return PredictorOutput(
            predictions=predictions,
            obs_component=obs_component,
            propio_component=propio_component,
            prior_mus=pred_output.prior_mus,
            prior_vars=pred_output.prior_vars,
            prior_logits=pred_output.prior_logits,
            priors=pred_output.priors,
            posterior_mus=pred_output.posterior_mus,
            posterior_vars=pred_output.posterior_vars,
            posterior_logits=pred_output.posterior_logits,
            posteriors=pred_output.posteriors,
        )

    def __getattr__(self, name):
        return getattr(self._predictor, name)


class _HierarchicalL1Wrapper:
    """
    JEPA-compatible facade for planners. Uses hierarchical prior under the hood
    but preserves the interface expected by existing planning code.
    """

    def __init__(self, hjepa: "HJEPAFeedback"):
        self._hier_model = hjepa
        self._level1 = hjepa.level1
        self.config = self._level1.config
        self.backbone = self._level1.backbone
        self.predictor = _HierarchicalPredictorProxy(hjepa)
        self.spatial_repr_dim = self._level1.spatial_repr_dim
        self.use_propio_pos = self._level1.use_propio_pos
        self.use_propio_vel = self._level1.use_propio_vel

    def forward_prior(self, *args, **kwargs):
        kwargs.pop("level", None)
        return self._hier_model.forward_prior(*args, **kwargs, level="l1").level1

    def forward_posterior(self, *args, **kwargs):
        return self._hier_model.forward_posterior(*args, **kwargs).level1

    def update_ema(self):
        self._hier_model.update_ema()

    def parameters(self, recurse: bool = True):
        return self._hier_model.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        self._hier_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self._hier_model.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        return self.to(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._level1, name)


class HJEPAFeedback(torch.nn.Module):
    """Hierarchical JEPA with cross-level feedback and slower L2 updates."""

    def __init__(
        self,
        config: HJEPAConfig,
        input_dim,
        normalizer=None,
        use_propio_pos=False,
        use_propio_vel=False,
    ):
        super().__init__()
        self.config = config
        self.normalizer = normalizer

        self.level1 = JEPA(
            config.level1,
            input_dim=input_dim,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

        self.l2_backbone = None
        self.l2_predictor = None
        self.l2_repr_dim = None
        self.l2_backbone_config = None
        self.l2_to_l1 = None
        self.level1_hierarchical = None

        if not self.config.disable_l2:
            if self.config.l2_action_agg != "concat":
                raise NotImplementedError(
                    f"Unknown l2_action_agg: {self.config.l2_action_agg}"
                )
            if self.config.l2_backbone is None:
                raise ValueError("l2_backbone is required for feedback hierarchy")

            if isinstance(self.config.l2_backbone, dict):
                self.l2_backbone_config = BackboneConfig.parse_from_dict(
                    self.config.l2_backbone
                )
            else:
                self.l2_backbone_config = self.config.l2_backbone

            l1_repr_dim = self.level1.repr_dim
            l2_input_dim = l1_repr_dim * max(1, self.config.step_skip)
            self.l2_backbone = build_backbone(
                self.l2_backbone_config,
                input_dim=l2_input_dim,
            )

            l2_spatial_repr_dim = self.l2_backbone.output_dim
            if isinstance(l2_spatial_repr_dim, tuple):
                l2_repr_dim = 1
                for dim in l2_spatial_repr_dim:
                    l2_repr_dim *= dim
            else:
                l2_repr_dim = l2_spatial_repr_dim
            self.l2_repr_dim = l2_repr_dim

            if self.config.l2_predictor is None:
                l2_predictor_cfg = dataclasses.replace(self.config.level1.predictor)
            elif isinstance(self.config.l2_predictor, dict):
                l2_predictor_cfg = PredictorConfig.parse_from_dict(
                    self.config.l2_predictor
                )
            else:
                l2_predictor_cfg = self.config.l2_predictor

            l2_action_dim = (
                self.config.level1.action_dim * max(1, self.config.step_skip)
                if self.config.l2_use_actions
                else 0
            )

            backbone_ln = getattr(self.l2_backbone, "final_ln", None)
            if l2_predictor_cfg.tie_backbone_ln and backbone_ln is None:
                backbone_ln = torch.nn.Identity()

            self.l2_predictor = build_predictor(
                l2_predictor_cfg,
                repr_dim=self.l2_repr_dim,
                action_dim=l2_action_dim,
                pred_propio_dim=self.l2_backbone.output_propio_dim,
                pred_obs_dim=self.l2_backbone.output_obs_dim,
                backbone_ln=backbone_ln,
            )

            if l1_repr_dim != self.l2_repr_dim:
                self.l2_to_l1 = torch.nn.Linear(self.l2_repr_dim, l1_repr_dim)
            else:
                self.l2_to_l1 = torch.nn.Identity()

            self.level1_hierarchical = _HierarchicalL1Wrapper(self)

    def _build_propio_states(self, propio_pos, propio_vel):
        if propio_pos is None or propio_vel is None:
            raise ValueError("propio_pos and propio_vel are required for proprio input")
        if propio_pos.numel() == 0:
            return propio_vel
        if propio_vel.numel() == 0:
            return propio_pos
        return torch.cat([propio_pos, propio_vel], dim=-1)

    def _build_l2_actions(self, actions: torch.Tensor, expected_steps: int):
        if actions is None:
            raise ValueError("actions are required when l2_use_actions=True")
        if expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if actions.shape[0] == expected_steps:
            return actions
        required_actions = expected_steps * max(1, self.config.step_skip)
        if actions.shape[0] < required_actions:
            raise ValueError("Not enough actions for L2 rollout")
        actions_trimmed = actions[:required_actions]
        action_chunks = []
        for i in range(expected_steps):
            start = i * self.config.step_skip
            end = (i + 1) * self.config.step_skip
            chunk = actions_trimmed[start:end]
            chunk = chunk.permute(1, 0, 2).reshape(actions.shape[1], -1)
            action_chunks.append(chunk)
        return torch.stack(action_chunks, dim=0)

    def _predict_next(self, predictor, current_state, action, rnn_state):
        if hasattr(predictor, "_is_rnn") and predictor._is_rnn():
            if rnn_state is None:
                rnn_state = current_state.unsqueeze(0).repeat(
                    predictor.num_layers, 1, 1
                )
            next_state, next_hidden_state = predictor.forward(
                rnn_state=rnn_state, rnn_input=action
            )
            return next_state, next_hidden_state
        return predictor.forward(current_state, action), None

    def _rollout(self, l1_init: torch.Tensor, actions: torch.Tensor, T: int):
        """posterior / prior 共通のオートレグレッシブロールアウト。

        Args:
            l1_init: 初期 L1 状態 (B, D)。backbone encoding of first observation。
            actions: (T, B, A)。
            T: 予測ステップ数。

        Returns:
            l1_preds: List[Tensor]。長さ T+1。[l1_init, pred_1, pred_2, ...]
            l2_encs:  List[Tensor]。L2 backbone 出力（L2 損失のターゲット）
            l2_preds: List[Tensor]。L2 predictor 出力（L2 損失の予測）
            l2_actions: Tensor or None。L2 用にチャンクしたアクション
        """
        step_skip = max(1, self.config.step_skip)
        l1_state = l1_init
        l1_preds = [l1_state]
        l1_history = [l1_state]
        l1_rnn_state = None

        # L2 初期化
        l2_feedback = torch.zeros_like(l1_init)
        l2_prev = None
        l2_encs = []
        l2_preds = []
        l2_rnn_state = None
        l2_actions = None
        l2_step_idx = 0
        l2_steps = 0

        if not self.config.disable_l2:
            l2_steps = T // step_skip
            if l2_steps > 0 and self.config.l2_use_actions:
                l2_actions = self._build_l2_actions(actions, l2_steps)

        for t in range(T):
            l1_input = l1_state + l2_feedback

            # RNN の場合、l2_feedback を hidden state に注入
            # MLP: forward(l1_state + l2_feedback, action) → 入力に含まれる
            # RNN: forward(rnn_state, action) → hidden state に加算して注入
            if (
                l1_rnn_state is not None
                and torch.is_tensor(l2_feedback)
                and l2_feedback.any()
            ):
                l1_rnn_state = l1_rnn_state + l2_feedback.unsqueeze(0)

            l1_next, l1_rnn_state = self._predict_next(
                self.level1.predictor, l1_input, actions[t], l1_rnn_state
            )
            l1_state = l1_next
            l1_preds.append(l1_next)
            l1_history.append(l1_next)

            # L2: step_skip 個の L1 予測が溜まったら L2 backbone + L2 predictor
            if not self.config.disable_l2 and (t + 1) % step_skip == 0:
                window = torch.stack(l1_history[-step_skip:], dim=0)
                l2_raw = window.permute(1, 0, 2).reshape(l1_init.shape[0], -1)
                l2_enc = self.l2_backbone.forward_multiple(
                    l2_raw.unsqueeze(0)
                ).encodings[0]
                l2_enc = flatten_conv_output(l2_enc)
                l2_encs.append(l2_enc)

                if l2_prev is None:
                    l2_prev = l2_enc
                    l2_preds.append(l2_enc)

                if l2_actions is not None and l2_step_idx < l2_steps:
                    l2_pred_input = l2_prev + l2_enc
                    l2_next, l2_rnn_state = self._predict_next(
                        self.l2_predictor,
                        l2_pred_input,
                        l2_actions[l2_step_idx],
                        l2_rnn_state,
                    )
                    l2_preds.append(l2_next)
                    l2_prev = l2_next
                    l2_feedback = self.l2_to_l1(l2_next)
                    l2_step_idx += 1

        return l1_preds, l2_encs, l2_preds, l2_actions

    def _encode_l1_prior(
        self,
        input_states: torch.Tensor,
        repr_input: bool,
        propio_pos: Optional[torch.Tensor],
        propio_vel: Optional[torch.Tensor],
    ):
        if repr_input:
            l1_state = input_states
        else:
            if self.level1.config.backbone.propio_dim is not None:
                propio_states = self._build_propio_states(propio_pos, propio_vel)
                l1_state = self.level1.backbone.forward_multiple(
                    input_states, propio=propio_states
                ).encodings
            else:
                l1_state = self.level1.backbone.forward_multiple(
                    input_states
                ).encodings

        l1_state = flatten_conv_output(l1_state)
        if l1_state.dim() == 3:
            l1_state = l1_state[0]
        if l1_state.dim() != 2:
            raise ValueError(
                f"Expected (B, D) L1 prior state, got {tuple(l1_state.shape)}"
            )
        return l1_state

    def _split_obs_and_propio(self, predictions: torch.Tensor, pred_propio_dim):
        if pred_propio_dim:
            if isinstance(pred_propio_dim, int):
                obs_component = predictions[:, :, :-pred_propio_dim]
                propio_component = predictions[:, :, -pred_propio_dim:]
            else:
                pred_propio_channels = pred_propio_dim[0]
                obs_component = predictions[:, :, :-pred_propio_channels]
                propio_component = predictions[:, :, -pred_propio_channels:]
        else:
            obs_component = predictions
            propio_component = None
        return obs_component, propio_component

    def forward_posterior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
        chunked_locations: Optional[torch.Tensor] = None,
        chunked_propio_pos: Optional[torch.Tensor] = None,
        chunked_propio_vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        # GT encodings（損失ターゲット用のみ）
        if self.level1.config.backbone.propio_dim is not None:
            propio_states = self._build_propio_states(propio_pos, propio_vel)
            l1_backbone_output = self.level1.backbone.forward_multiple(
                input_states, propio=propio_states
            )
        else:
            l1_backbone_output = self.level1.backbone.forward_multiple(input_states)

        l1_encs = flatten_conv_output(l1_backbone_output.encodings)

        # 共通ロールアウト（prior と完全に同じ計算グラフ）
        T = input_states.shape[0] - 1
        l1_preds, l2_encs, l2_preds, l2_actions = self._rollout(
            l1_encs[0], actions, T
        )

        # L1 結果パッケージ（backbone_output は素の GT encodings）
        l1_result = JEPAForwardResult(
            backbone_output=BackboneOutput(encodings=l1_encs),
            ema_backbone_output=None,
            pred_output=PredictorOutput(predictions=torch.stack(l1_preds)),
            actions=actions,
        )

        # L2 結果パッケージ
        # _rollout は N 個の l2_encs, N+1 個の l2_preds を返す。
        # l2_preds[:N] に切り詰める（最後の予測にターゲットがないため）。
        l2_result = None
        if l2_encs:
            n = len(l2_encs)
            l2_result = JEPAForwardResult(
                backbone_output=BackboneOutput(
                    encodings=torch.stack(l2_encs)
                ),
                ema_backbone_output=None,
                pred_output=PredictorOutput(
                    predictions=torch.stack(l2_preds[:n])
                ),
                actions=l2_actions,
            )

        return ForwardResult(level1=l1_result, level2=l2_result)

    def forward_prior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        *,
        repr_input: bool = False,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        level: str = "l1",
    ) -> ForwardResult:
        if level not in ("l1", "l2"):
            raise ValueError(f"Unknown level: {level}")

        if level == "l2" and self.config.disable_l2:
            raise RuntimeError("L2 is disabled")

        # Latent-action shortcut: delegate to L1 only
        if latents is not None:
            if level == "l2":
                raise NotImplementedError(
                    "level='l2' with latents is unsupported"
                )
            result = self.level1.forward_prior(
                input_states=input_states,
                actions=actions,
                T=T,
                repr_input=repr_input,
                propio_pos=propio_pos,
                propio_vel=propio_vel,
                latents=latents,
                goal=goal,
            )
            return ForwardResult(level1=result, level2=None)

        # Determine rollout horizon
        if T is None:
            if actions is None:
                raise ValueError("T is None but actions are not provided")
            T = actions.shape[0]

        if T < 0:
            raise ValueError("T must be non-negative")
        if T > 0 and actions is None:
            raise ValueError(
                "actions are required for prior rollout when T > 0"
            )

        actions_rollout = actions
        if actions_rollout is not None:
            if actions_rollout.shape[0] < T:
                raise ValueError(
                    "Not enough actions for requested rollout horizon"
                )
            actions_rollout = actions_rollout[:T]

        # 初期状態エンコード
        l1_init = self._encode_l1_prior(
            input_states=input_states,
            repr_input=repr_input,
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )

        # 共通ロールアウト（posterior と同じ計算グラフ）
        l1_preds, l2_encs, l2_preds, l2_actions = self._rollout(
            l1_init, actions_rollout, T
        )

        # L1 結果パッケージ
        l1_preds_t = torch.stack(l1_preds)
        l1_obs, l1_propio = self._split_obs_and_propio(
            l1_preds_t, self.level1.predictor.pred_propio_dim
        )
        l1_result = JEPAForwardResult(
            backbone_output=None,
            ema_backbone_output=None,
            pred_output=PredictorOutput(
                predictions=l1_preds_t,
                obs_component=l1_obs,
                propio_component=l1_propio,
            ),
            actions=actions_rollout,
        )

        # L2 結果パッケージ
        l2_result = None
        if l2_preds:
            l2_preds_t = torch.stack(l2_preds)
            l2_obs, l2_propio = self._split_obs_and_propio(
                l2_preds_t, self.l2_predictor.pred_propio_dim
            )
            l2_result = JEPAForwardResult(
                backbone_output=None,
                ema_backbone_output=None,
                pred_output=PredictorOutput(
                    predictions=l2_preds_t,
                    obs_component=l2_obs,
                    propio_component=l2_propio,
                ),
                actions=l2_actions,
            )

        if level == "l2":
            return ForwardResult(level1=None, level2=l2_result)
        return ForwardResult(level1=l1_result, level2=l2_result)

    def update_ema(self):
        self.level1.update_ema()
