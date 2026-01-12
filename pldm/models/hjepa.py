from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import dataclasses
from functools import reduce
import operator

import torch
import random
from torch.distributions import Normal
from torch.nn import functional as F

from pldm.configs import ConfigBase
from pldm.models.encoders.encoders import build_backbone
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.enums import PredictorOutput
from pldm.models.jepa import JEPA, JEPAConfig, ForwardResult as JEPAForwardResult
from pldm.models.misc import build_mlp
from pldm.models.predictors import RSSMPredictor
from pldm.models.utils import flatten_conv_output


@dataclass
class HJEPAConfig(ConfigBase):
    # L1(低レベル)とL2(高レベル)をまとめた設定
    level1: JEPAConfig = field(default_factory=JEPAConfig)
    step_skip: int = 4
    disable_l2: bool = False
    freeze_l1: bool = False
    train_l1: bool = False
    l1_n_steps: int = 17
    l2_backbone: Optional[BackboneConfig] = None
    l2_z_dim: int = 32
    l2_rnn_state_dim: Optional[int] = None
    l2_min_var: float = 0.1
    l2_use_action_only: bool = False
    l2_posterior_arch: str = "512-512"
    l2_decoder_arch: str = "512-512"
    l2_use_actions: bool = True
    l2_action_agg: str = "concat"
    l2_condition_l1: bool = False
    l2_subgoal: bool = False
    l2_subgoal_coeff: float = 1.0


class ForwardResult(NamedTuple):
    level1: Optional[JEPAForwardResult]
    level2: Optional[JEPAForwardResult] = None


class _L1ConditionedWrapper:
    def __init__(self, hjepa: "HJEPA"):
        self._hjepa = hjepa
        self._level1 = hjepa.level1
        self.config = self._level1.config
        self.backbone = self._level1.backbone
        self.predictor = self._level1.predictor
        self.spatial_repr_dim = self._level1.spatial_repr_dim
        self.use_propio_pos = self._level1.use_propio_pos
        self.use_propio_vel = self._level1.use_propio_vel

    def forward_prior(self, *args, **kwargs):
        kwargs.pop("level", None)
        return self._hjepa.forward_prior(*args, **kwargs, level="l1").level1

    def forward_posterior(self, *args, **kwargs):
        return self._hjepa.forward_posterior(*args, **kwargs).level1

    def update_ema(self):
        self._level1.update_ema()

    def parameters(self, recurse: bool = True):
        return self._level1.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        self._level1.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self._level1.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        return self.to(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._level1, name)


class HJEPA(torch.nn.Module): # L1はjepa.pyと同じ、l2はRSSMでprior/posterior予測、Decoderで次の表現予測
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
        # L1は通常のJEPAをそのまま使う
        self.level1 = JEPA(
            config.level1,
            input_dim=input_dim,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

        self.normalizer = normalizer
        self.l2_repr_dim = None
        self.l2_rnn_state_dim = None
        self.l2_backbone = None
        self.l2_backbone_config = None
        self.l2_action_projector = None
        self.level1_conditioned = None
        if not self.config.disable_l2:
            # L2用のバックボーン設定（未指定ならL1をコピー）
            if self.config.l2_backbone is None:
                self.l2_backbone_config = dataclasses.replace(
                    self.config.level1.backbone
                )
            elif isinstance(self.config.l2_backbone, dict):
                self.l2_backbone_config = BackboneConfig.parse_from_dict(
                    self.config.l2_backbone
                )
            else:
                self.l2_backbone_config = self.config.l2_backbone

            # L2は別バックボーンで表現を作る
            self.l2_backbone = build_backbone(
                self.l2_backbone_config,
                input_dim=input_dim,
            )
            l2_spatial_repr_dim = self.l2_backbone.output_dim
            if isinstance(l2_spatial_repr_dim, tuple):
                self.l2_repr_dim = reduce(operator.mul, l2_spatial_repr_dim)
            else:
                self.l2_repr_dim = l2_spatial_repr_dim
            # L2のRNN状態次元（未指定なら表現次元）
            self.l2_rnn_state_dim = (
                self.config.l2_rnn_state_dim
                if self.config.l2_rnn_state_dim is not None
                else self.l2_repr_dim
            )
            if self.config.l2_use_action_only and not self.config.l2_use_actions:
                raise ValueError("l2_use_action_only requires l2_use_actions")
            # L2はstep_skip分のアクションをまとめて使う
            l2_action_dim = (
                self.level1.config.action_dim * self.config.step_skip
                if self.config.l2_use_actions
                else 0
            )
            if self.config.l2_use_action_only and l2_action_dim == 0:
                raise ValueError("l2_use_action_only requires non-zero action_dim")
            # RSSMの予測器（prior）とposterior/decoder
            self.predictor_l2 = RSSMPredictor(
                rnn_state_dim=self.l2_rnn_state_dim,
                z_dim=self.config.l2_z_dim,
                action_dim=l2_action_dim,
                min_var=self.config.l2_min_var,
                use_action_only=self.config.l2_use_action_only,
            )
            posterior_input_dim = self.l2_repr_dim + self.l2_rnn_state_dim
            # l2のposteriorとdecoderはシンプルなmlp
            self.posterior_l2 = build_mlp(
                self.config.l2_posterior_arch,
                input_dim=posterior_input_dim,
                output_shape=self.config.l2_z_dim * 2,
            )
            decoder_input_dim = self.l2_rnn_state_dim + self.config.l2_z_dim
            self.decoder = build_mlp(
                self.config.l2_decoder_arch,
                input_dim=decoder_input_dim,
                output_shape=self.l2_repr_dim,
            )
            # L2表現とRNN状態次元が違う場合は射影(線形層で次元を揃える)
            if self.l2_rnn_state_dim != self.l2_repr_dim:
                self.l2_state_projector = torch.nn.Linear(
                    self.l2_repr_dim, self.l2_rnn_state_dim
                )
            else:
                self.l2_state_projector = torch.nn.Identity()

            if self.config.l2_condition_l1:
                if self.level1.config.action_dim <= 0:
                    raise ValueError("L2 conditioning requires L1 action_dim > 0")
                self.l2_action_projector = torch.nn.Linear(
                    self.config.l2_z_dim, self.level1.config.action_dim
                )
                self.level1_conditioned = _L1ConditionedWrapper(self)

    def _build_propio_states(self, propio_pos, propio_vel): # propio:ロボットのアームの関節位置・速度
        # propio_pos/vel を結合してバックボーンに渡す
        if propio_pos is None or propio_vel is None:
            raise ValueError("propio_pos and propio_vel are required for proprio input")
        if propio_pos.numel() == 0:
            return propio_vel
        if propio_vel.numel() == 0:
            return propio_pos
        return torch.cat([propio_pos, propio_vel], dim=-1)

    def _encode_l2(self, input_states, propio_pos=None, propio_vel=None):
        # L2バックボーンで入力を符号化
        if self.l2_backbone is None:
            raise RuntimeError("L2 backbone is not initialized")
        if self.l2_backbone_config.propio_dim is not None:
            propio_states = self._build_propio_states(propio_pos, propio_vel)
            return self.l2_backbone.forward_multiple(input_states, propio=propio_states)
        return self.l2_backbone.forward_multiple(input_states)

    def _build_l2_actions(self, actions, expected_steps):
        # step_skip分のアクションをまとめてL2の1ステップにする
        if actions is None:
            raise ValueError("actions are required when l2_use_actions=True")
        if expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if actions.shape[0] == expected_steps:
            return actions
        required_actions = expected_steps * self.config.step_skip
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

    def _expand_l2_latents(self, latents: torch.Tensor, target_steps: int):
        if latents is None:
            return None
        expanded = latents.repeat_interleave(self.config.step_skip, dim=0)
        if expanded.shape[0] < target_steps:
            pad = expanded[-1:].repeat(target_steps - expanded.shape[0], 1, 1)
            expanded = torch.cat([expanded, pad], dim=0)
        if expanded.shape[0] > target_steps:
            expanded = expanded[:target_steps]
        return expanded

    def _condition_l1_actions(self, actions, l2_latents, target_steps: int):
        if (
            actions is None
            or l2_latents is None
            or self.l2_action_projector is None
        ):
            return actions
        expanded_latents = self._expand_l2_latents(l2_latents, target_steps)
        cond_actions = self.l2_action_projector(expanded_latents)
        return actions + cond_actions

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
        actions_are_l2: bool = False,
        level: str = "l1",
    ) -> ForwardResult:
        # levelでL1/L2のpriorロールアウトを切り替える
        if level == "l1":
            conditioned_actions = actions
            if (
                self.config.l2_condition_l1
                and not self.config.disable_l2
                and not repr_input
                and actions is not None
            ):
                l2_result = self._forward_prior_l2(
                    input_states=input_states,
                    actions=actions,
                    T=T,
                    repr_input=repr_input,
                    propio_pos=propio_pos,
                    propio_vel=propio_vel,
                    latents=None,
                    actions_are_l2=actions_are_l2,
                )
                l2_latents = (
                    l2_result.pred_output.priors
                    if l2_result is not None and l2_result.pred_output is not None
                    else None
                )
                conditioned_actions = self._condition_l1_actions(
                    actions,
                    l2_latents,
                    target_steps=actions.shape[0],
                )

            return ForwardResult(
                level1=self.level1.forward_prior(
                    input_states,
                    repr_input=repr_input,
                    actions=conditioned_actions,
                    propio_pos=propio_pos,
                    propio_vel=propio_vel,
                    latents=latents,
                    goal=goal,
                    T=T,
                )
            )
        if level != "l2":
            raise ValueError(f"Unknown level: {level}")
        if self.config.disable_l2:
            raise RuntimeError("L2 is disabled")

        # L2のpriorロールアウト
        l2_result = self._forward_prior_l2(
            input_states=input_states,
            actions=actions,
            T=T,
            repr_input=repr_input,
            propio_pos=propio_pos,
            propio_vel=propio_vel,
            latents=latents,
            actions_are_l2=actions_are_l2,
        )
        return ForwardResult(level1=None, level2=l2_result)

    def _forward_prior_l2(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor],
        T: Optional[int],
        repr_input: bool,
        propio_pos: Optional[torch.Tensor],
        propio_vel: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        actions_are_l2: bool,
    ) -> JEPAForwardResult:
        # 入力が表現か画像かで処理を分岐
        if repr_input: # すでに表現の場合
            current_state = input_states
        else: # 画像入力ならバックボーンで表現化
            l2_encode_result = self._encode_l2(
                input_states, propio_pos=propio_pos, propio_vel=propio_vel
            )
            current_state = l2_encode_result.encodings

        current_state = flatten_conv_output(current_state)
        if current_state.dim() == 3:
            current_state = current_state[0]

        # 予測ステップ数を決定（T or actions/latents）
        if T is None:
            if latents is not None:
                expected_steps = latents.shape[0]
            elif actions is not None:
                if actions_are_l2:
                    expected_steps = actions.shape[0]
                else:
                    expected_steps = actions.shape[0] // self.config.step_skip
                    if expected_steps < 1:
                        raise ValueError("Not enough actions for L2 rollout")
            else:
                raise ValueError("T is None but actions and latents are None")
        else:
            expected_steps = T

        if latents is not None and latents.shape[0] != expected_steps:
            raise ValueError("latents length does not match expected_steps")

        actions_l2 = None
        if self.config.l2_use_actions:
            if self.config.l2_action_agg != "concat":
                raise NotImplementedError(
                    f"Unknown l2_action_agg: {self.config.l2_action_agg}"
                )
            actions_l2 = self._build_l2_actions(actions, expected_steps)

        # RNN初期状態を作ってpriorロールアウト
        rnn_state = self.l2_state_projector(current_state)
        batch_size = rnn_state.shape[0]
        sampled_z = torch.zeros(
            batch_size, self.config.l2_z_dim, device=rnn_state.device
        )

        preds = [current_state]
        prior_mus = []
        prior_vars = []
        priors = []

        for i in range(expected_steps):
            action_i = actions_l2[i] if actions_l2 is not None else None
            input_z = latents[i] if latents is not None else sampled_z
            rnn_state, prior_mu, prior_var = self.predictor_l2(
                input_z, action_i, rnn_state
            )
            prior_var = F.softplus(prior_var) + self.config.l2_min_var
            prior_mus.append(prior_mu)
            prior_vars.append(prior_var)

            if latents is None:
                prior_dist = Normal(prior_mu, prior_var)
                sampled_z = prior_dist.rsample()
            else:
                sampled_z = latents[i]
            priors.append(sampled_z)

            # 予測表現をデコード
            pred = self.decoder(torch.cat([rnn_state, sampled_z], dim=-1))
            preds.append(pred)

        preds = torch.stack(preds, dim=0)
        prior_mus = torch.stack(prior_mus) if prior_mus else None
        prior_vars = torch.stack(prior_vars) if prior_vars else None
        priors = torch.stack(priors) if priors else None

        pred_output_l2 = PredictorOutput(
            predictions=preds,
            prior_mus=prior_mus,
            prior_vars=prior_vars,
            priors=priors,
        )
        return JEPAForwardResult(
            backbone_output=None,
            ema_backbone_output=None,
            pred_output=pred_output_l2,
            actions=actions_l2,
        )

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
        forward_result_l1 = None
        forward_result_l2 = None
        l2_posteriors = None

        if not self.config.disable_l2:
            # L2はstep_skipごとに間引いた表現で学習
            l2_encode_result = self._encode_l2(
                input_states, propio_pos=propio_pos, propio_vel=propio_vel
            )
            encs = flatten_conv_output(l2_encode_result.encodings)
            step_skip = self.config.step_skip
            max_steps = (encs.shape[0] - 1) // step_skip
            if max_steps < 1:
                raise ValueError("Not enough steps for L2 rollout")
            encs_l2 = encs[::step_skip][: max_steps + 1]

            actions_l2 = None
            if self.config.l2_use_actions:
                if self.config.l2_action_agg != "concat":
                    raise NotImplementedError(
                        f"Unknown l2_action_agg: {self.config.l2_action_agg}"
                    )
                actions_l2 = self._build_l2_actions(actions, max_steps)

            rnn_state = self.l2_state_projector(encs_l2[0])
            batch_size = rnn_state.shape[0]
            sampled_z = torch.zeros(
                batch_size, self.config.l2_z_dim, device=encs_l2.device
            )

            preds = [encs_l2[0]]
            prior_mus = []
            prior_vars = []
            priors = []
            posterior_mus = []
            posterior_vars = []
            posteriors = []

            for i in range(max_steps):
                action_i = actions_l2[i] if actions_l2 is not None else None
                rnn_state, prior_mu, prior_var = self.predictor_l2(
                    sampled_z, action_i, rnn_state
                )
                prior_var = F.softplus(prior_var) + self.config.l2_min_var
                prior_mus.append(prior_mu)
                prior_vars.append(prior_var)
                prior_dist = Normal(prior_mu, prior_var)
                priors.append(prior_dist.rsample())

                # posteriorは次時刻の表現と現在のRNN状態から推定
                posterior_input = torch.cat([encs_l2[i + 1], rnn_state], dim=-1)
                posterior_stats = self.posterior_l2(posterior_input)
                posterior_mu, posterior_var = torch.chunk(posterior_stats, 2, dim=-1)
                posterior_var = F.softplus(posterior_var) + self.config.l2_min_var
                posterior_mus.append(posterior_mu)
                posterior_vars.append(posterior_var)
                posterior_dist = Normal(posterior_mu, posterior_var)
                sampled_z = posterior_dist.rsample()
                posteriors.append(sampled_z)

                # posteriorから表現をデコード
                pred = self.decoder(torch.cat([rnn_state, sampled_z], dim=-1))
                preds.append(pred)

            preds = torch.stack(preds, dim=0)
            prior_mus = torch.stack(prior_mus) if prior_mus else None
            prior_vars = torch.stack(prior_vars) if prior_vars else None
            priors = torch.stack(priors) if priors else None
            posterior_mus = torch.stack(posterior_mus) if posterior_mus else None
            posterior_vars = torch.stack(posterior_vars) if posterior_vars else None
            posteriors = torch.stack(posteriors) if posteriors else None

            backbone_output_l2 = BackboneOutput(encodings=encs_l2)
            ema_backbone_output_l2 = None
            pred_output_l2 = PredictorOutput(
                predictions=preds,
                prior_mus=prior_mus,
                prior_vars=prior_vars,
                priors=priors,
                posterior_mus=posterior_mus,
                posterior_vars=posterior_vars,
                posteriors=posteriors,
            )
            forward_result_l2 = JEPAForwardResult(
                backbone_output=backbone_output_l2,
                ema_backbone_output=ema_backbone_output_l2,
                pred_output=pred_output_l2,
                actions=actions_l2,
            )
            l2_posteriors = pred_output_l2.posteriors

        if self.config.train_l1:
            # L1は部分系列を抜き出して学習
            if self.config.l2_subgoal:
                sub_idx = 0
            else:
                sub_idx = random.randint(
                    0, input_states.shape[0] - self.config.l1_n_steps
                )
            l1_input_states = input_states[sub_idx : sub_idx + self.config.l1_n_steps]
            l1_actions = actions[sub_idx : sub_idx + self.config.l1_n_steps - 1]

            if (
                self.config.l2_condition_l1
                and l2_posteriors is not None
                and actions is not None
            ):
                conditioned_actions = self._condition_l1_actions(
                    actions,
                    l2_posteriors,
                    target_steps=actions.shape[0],
                )
                l1_actions = conditioned_actions[
                    sub_idx : sub_idx + self.config.l1_n_steps - 1
                ]

            forward_result_l1 = self.level1.forward_posterior(
                l1_input_states,
                l1_actions,
                propio_pos=propio_pos,
                propio_vel=propio_vel,
                chunked_locations=chunked_locations,
                chunked_propio_pos=chunked_propio_pos,
                chunked_propio_vel=chunked_propio_vel,
                encode_only=False,
            )

        return ForwardResult(level1=forward_result_l1, level2=forward_result_l2)

    def update_ema(self):
        # L1のEMAのみ更新
        self.level1.update_ema()
