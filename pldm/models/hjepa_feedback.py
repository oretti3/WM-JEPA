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
        self.l1_to_l2 = None
        self.l2_to_l1 = None

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
                self.l1_to_l2 = torch.nn.Linear(l1_repr_dim, self.l2_repr_dim)
                self.l2_to_l1 = torch.nn.Linear(self.l2_repr_dim, l1_repr_dim)
            else:
                self.l1_to_l2 = torch.nn.Identity()
                self.l2_to_l1 = torch.nn.Identity()

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

    def _build_l2_inputs(self, l1_encs_flat: torch.Tensor):
        step_skip = max(1, self.config.step_skip)
        max_steps = (l1_encs_flat.shape[0] - 1) // step_skip
        if max_steps < 1:
            raise ValueError("Not enough steps for L2 rollout")
        l2_inputs = []
        for i in range(max_steps + 1):
            start = i * step_skip
            end = min(start + step_skip, l1_encs_flat.shape[0])
            chunk = l1_encs_flat[start:end]
            if chunk.shape[0] < step_skip:
                pad = chunk[-1:].repeat(step_skip - chunk.shape[0], 1, 1)
                chunk = torch.cat([chunk, pad], dim=0)
            chunk = chunk.permute(1, 0, 2).reshape(l1_encs_flat.shape[1], -1)
            l2_inputs.append(chunk)
        return torch.stack(l2_inputs, dim=0)

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
        if self.level1.config.backbone.propio_dim is not None:
            propio_states = self._build_propio_states(propio_pos, propio_vel)
            l1_backbone_output = self.level1.backbone.forward_multiple(
                input_states, propio=propio_states
            )
        else:
            l1_backbone_output = self.level1.backbone.forward_multiple(input_states)

        l1_encs = flatten_conv_output(l1_backbone_output.encodings)
        if l1_encs.dim() == 3:
            l1_encs = l1_encs
        else:
            raise ValueError("Feedback hierarchy expects flat L1 encodings")

        ema_backbone_output = None
        if self.level1.backbone_ema is not None:
            if self.level1.config.backbone.propio_dim is not None:
                propio_states = self._build_propio_states(propio_pos, propio_vel)
                ema_backbone_output = self.level1.backbone_ema.forward_multiple(
                    input_states, propio=propio_states
                )
            else:
                ema_backbone_output = self.level1.backbone_ema.forward_multiple(
                    input_states
                )

        l2_backbone_output = None
        l2_preds = None
        l2_actions = None
        l2_encs_aug = None

        l2_prev = None
        l2_rnn_state = None
        if not self.config.disable_l2:
            l2_inputs = self._build_l2_inputs(l1_encs)
            l2_backbone_output = self.l2_backbone.forward_multiple(l2_inputs)
            l2_encs = flatten_conv_output(l2_backbone_output.encodings)

            num_l2_steps = l2_encs.shape[0] - 1
            if self.config.l2_use_actions:
                l2_actions = self._build_l2_actions(actions, num_l2_steps)

            l2_prev = torch.zeros_like(l2_encs[0])
            l2_preds = []
            l2_encs_aug = []

        l1_prev = torch.zeros_like(l1_encs[0])
        l1_rnn_state = None
        l1_preds = []
        l1_encs_aug = []
        feedback_terms = []

        l2_index = 0
        for t in range(l1_encs.shape[0]):
            if l2_prev is not None:
                l2_feedback = self.l2_to_l1(l2_prev)
            else:
                l2_feedback = 0.0

            l1_input = l1_encs[t] + l1_prev + l2_feedback
            l1_encs_aug.append(l1_input)
            feedback_terms.append(l1_prev + l2_feedback)

            if t == 0:
                l1_preds.append(l1_input)

            if (
                l2_prev is not None
                and t % max(1, self.config.step_skip) == 0
                and l2_index < l2_encs.shape[0]
            ):
                l2_input = (
                    l2_encs[l2_index]
                    + l2_prev
                    + self.l1_to_l2(l1_prev)
                )
                l2_encs_aug.append(l2_input)
                if l2_index == 0:
                    l2_preds.append(l2_input)
                if l2_index < l2_encs.shape[0] - 1:
                    if l2_actions is None:
                        raise ValueError("l2_use_actions=False is unsupported here")
                    l2_next, l2_rnn_state = self._predict_next(
                        self.l2_predictor,
                        l2_input,
                        l2_actions[l2_index],
                        l2_rnn_state,
                    )
                    l2_preds.append(l2_next)
                    l2_prev = l2_next
                l2_index += 1

            if t < l1_encs.shape[0] - 1:
                if actions is None:
                    raise ValueError("actions are required for L1 rollout")
                l1_next, l1_rnn_state = self._predict_next(
                    self.level1.predictor,
                    l1_input,
                    actions[t],
                    l1_rnn_state,
                )
                l1_preds.append(l1_next)
                l1_prev = l1_next

        l1_encs_aug = torch.stack(l1_encs_aug, dim=0)
        l1_preds = torch.stack(l1_preds, dim=0)

        if ema_backbone_output is not None:
            ema_encs = flatten_conv_output(ema_backbone_output.encodings)
            ema_aug = []
            for t in range(ema_encs.shape[0]):
                ema_aug.append(ema_encs[t] + feedback_terms[t])
            ema_backbone_output = BackboneOutput(encodings=torch.stack(ema_aug, dim=0))

        l1_backbone_output = BackboneOutput(encodings=l1_encs_aug)
        l1_pred_output = PredictorOutput(predictions=l1_preds)
        forward_result_l1 = JEPAForwardResult(
            backbone_output=l1_backbone_output,
            ema_backbone_output=ema_backbone_output,
            pred_output=l1_pred_output,
            actions=actions,
        )

        forward_result_l2 = None
        if l2_preds is not None and l2_encs_aug is not None:
            l2_encs_aug = torch.stack(l2_encs_aug, dim=0)
            l2_preds = torch.stack(l2_preds, dim=0)
            l2_backbone_output = BackboneOutput(encodings=l2_encs_aug)
            l2_pred_output = PredictorOutput(predictions=l2_preds)
            forward_result_l2 = JEPAForwardResult(
                backbone_output=l2_backbone_output,
                ema_backbone_output=None,
                pred_output=l2_pred_output,
                actions=l2_actions,
            )

        return ForwardResult(level1=forward_result_l1, level2=forward_result_l2)

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
        if level != "l1":
            raise ValueError("Feedback hierarchy only supports level='l1' prior")
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

    def update_ema(self):
        self.level1.update_ema()
