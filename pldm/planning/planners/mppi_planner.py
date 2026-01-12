from typing import Optional, Callable

import torch
from torch import nn

from .mppi_torch import MPPI
from pldm.models.utils import flatten_conv_output
from .planner import PlanningResult


class LearnedDynamics:
    def __init__(self, model, state_dim=None):
        self.model = model
        self.dump_dict = None
        self.state_dim = state_dim
        self.max_batch_size = 500
        self.hjepa = self._resolve_hjepa(model)
        self.use_l2_conditioning = bool(
            self.hjepa
            and getattr(self.hjepa.config, "l2_condition_l1", False)
            and not getattr(self.hjepa.config, "disable_l2", True)
        )
        self.raw_obs = None
        self.raw_propio_pos = None
        self.raw_propio_vel = None
        self._conditioned_actions = None

    @staticmethod
    def _resolve_hjepa(model):
        if hasattr(model, "_hjepa"):
            return model._hjepa
        if hasattr(model, "config") and hasattr(model, "level1"):
            if hasattr(model.config, "l2_condition_l1"):
                return model
        return None

    def set_rollout_context(
        self,
        raw_obs: Optional[torch.Tensor] = None,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
    ):
        self.raw_obs = raw_obs
        self.raw_propio_pos = propio_pos
        self.raw_propio_vel = propio_vel

    def prepare_rollout(self, state, perturbed_actions):
        self._conditioned_actions = None

        if not self.use_l2_conditioning:
            return
        if self.raw_obs is None or perturbed_actions is None:
            return
        if not torch.is_tensor(perturbed_actions) or perturbed_actions.dim() != 3:
            return

        actions_l1 = perturbed_actions.permute(1, 0, 2).contiguous()
        self._conditioned_actions = self._compute_conditioned_actions(
            actions=actions_l1,
            raw_obs=self.raw_obs,
            propio_pos=self.raw_propio_pos,
            propio_vel=self.raw_propio_vel,
        )

    def _compute_conditioned_actions(
        self,
        actions: torch.Tensor,
        raw_obs: torch.Tensor,
        propio_pos: Optional[torch.Tensor] = None,
        propio_vel: Optional[torch.Tensor] = None,
    ):
        if not self.use_l2_conditioning:
            return None
        if actions is None or raw_obs is None:
            return None
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        if actions.dim() != 3:
            return None

        hjepa = self.hjepa
        if hjepa is None or getattr(hjepa, "l2_action_projector", None) is None:
            return None

        with torch.no_grad():
            device = next(hjepa.parameters()).device
            actions = actions.to(device)
            raw_obs = raw_obs if torch.is_tensor(raw_obs) else torch.as_tensor(raw_obs)
            raw_obs = raw_obs.to(device)
            if raw_obs.dim() in (1, 3):
                raw_obs = raw_obs.unsqueeze(0)

            batch_size = actions.shape[1]
            if raw_obs.shape[0] != batch_size:
                if raw_obs.shape[0] == 1:
                    raw_obs = raw_obs.expand(batch_size, *raw_obs.shape[1:])
                else:
                    raw_obs = raw_obs[:batch_size]

            if propio_pos is not None and not torch.is_tensor(propio_pos):
                propio_pos = torch.as_tensor(propio_pos)
            if propio_pos is not None:
                propio_pos = propio_pos.to(device)
                if propio_pos.dim() == 1:
                    propio_pos = propio_pos.unsqueeze(0)
                if propio_pos.shape[0] != batch_size:
                    if propio_pos.shape[0] == 1:
                        propio_pos = propio_pos.expand(batch_size, *propio_pos.shape[1:])
                    else:
                        propio_pos = propio_pos[:batch_size]

            if propio_vel is not None and not torch.is_tensor(propio_vel):
                propio_vel = torch.as_tensor(propio_vel)
            if propio_vel is not None:
                propio_vel = propio_vel.to(device)
                if propio_vel.dim() == 1:
                    propio_vel = propio_vel.unsqueeze(0)
                if propio_vel.shape[0] != batch_size:
                    if propio_vel.shape[0] == 1:
                        propio_vel = propio_vel.expand(batch_size, *propio_vel.shape[1:])
                    else:
                        propio_vel = propio_vel[:batch_size]

            try:
                l2_result = hjepa._forward_prior_l2(
                    input_states=raw_obs,
                    actions=actions,
                    T=None,
                    repr_input=False,
                    propio_pos=propio_pos,
                    propio_vel=propio_vel,
                    latents=None,
                    actions_are_l2=False,
                )
            except ValueError:
                return None

            l2_pred_output = getattr(l2_result, "pred_output", None)
            l2_latents = None if l2_pred_output is None else l2_pred_output.priors
            if l2_latents is None:
                return None

            return hjepa._condition_l1_actions(
                actions,
                l2_latents,
                target_steps=actions.shape[0],
            )

    def _match_action_shape(self, action, conditioned_action):
        if not torch.is_tensor(conditioned_action):
            return action
        cond = conditioned_action.to(action.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        flat_action = action.reshape(-1, action.shape[-1])
        flat_cond = cond
        if flat_cond.shape[0] != flat_action.shape[0]:
            if flat_action.shape[0] % flat_cond.shape[0] == 0:
                repeats = flat_action.shape[0] // flat_cond.shape[0]
            else:
                repeats = flat_action.shape[0] // flat_cond.shape[0] + 1
            flat_cond = flat_cond.repeat(repeats, 1)[: flat_action.shape[0]]
        flat_cond = flat_cond[: flat_action.shape[0]]
        return flat_cond.reshape(action.shape)

    def __call__(
        self, state, action, t=None, only_return_last=True, flatten_output=True
    ):
        """
        state: [K x nx]
        action: [K x nx]
        """
        if torch.is_tensor(t):
            t = int(t.item())
        if (
            self.use_l2_conditioning
            and self._conditioned_actions is not None
            and self.model.config.action_dim
            and t is not None
        ):
            if t < self._conditioned_actions.shape[0]:
                action = self._match_action_shape(
                    action, self._conditioned_actions[t]
                )

        # make sure state is in correct format
        og_shape = state.shape
        n_samples = og_shape[0]

        if isinstance(self.state_dim, int):
            self.state_dim = (self.state_dim,)

        new_shape = (n_samples, *self.state_dim)
        state = state.view(new_shape)

        # introduce time dimension to action if needed
        if len(action.shape) < 3:
            action = action.unsqueeze(0)

        T = action.shape[0]

        if self.model.config.action_dim:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                action.float(),
                T,
            )
        else:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                actions=None,
                T=T,
                latents=action.float(),
            )

        preds = pred_output.predictions
        pred_obs = pred_output.obs_component

        if flatten_output:
            preds = flatten_conv_output(preds)  # required for 3rd party MPPI code...
            pred_obs = flatten_conv_output(pred_obs)

        if only_return_last:
            preds = preds[-1]
            pred_obs = pred_obs[-1]

        # we need to return both. preds is used to propagate the state forward. pred_obs is used to take cost
        return preds, pred_obs

    def before_planning_callback(self):
        self.orig_training_state = self.model.training
        self.model.train(False)

    def after_planning_callback(self):
        self.model.train(self.orig_training_state)


class RunningCost:
    def __init__(self, objective, idx=None, projector=None):
        self.objective = objective
        self.idx = idx
        self.projector = nn.Identity() if projector is None else projector

    def __call__(self, state, action, t=None):
        """encoding shape is B X D
        Note that B are samples for the same environment
        You want to diff against target_enc of shape (D) retrieved from objective
        """
        objective = self.objective
        target = objective.target_enc[self.idx]

        state = self.projector(state)
        target = self.projector(target)

        diff = (state - target).pow(2)

        return diff.mean(dim=1)


class MPPIPlanner:
    def __init__(
        self,
        config,
        model,
        normalizer,
        objective,
        prober: Optional[torch.nn.Module] = None,
        action_normalizer: Optional[Callable] = None,
        num_refinement_steps: int = 1,
        n_envs: int = None,
        l2: bool = False,
        projected_cost: bool = False,
    ):
        device = next(model.parameters()).device

        latent_actions = l2 and model.config.predictor.z_dim > 0
        self.model = model
        self.config = config
        self.dynamics = LearnedDynamics(
            model,
            state_dim=model.spatial_repr_dim,
        )
        self.normalizer = normalizer
        self.action_normalizer = action_normalizer
        self.prober = prober
        self.latent_actions = latent_actions

        noise_sigma = torch.diag(
            torch.tensor(
                [config.noise_sigma] * model.predictor.action_dim,
                dtype=torch.float32,
                device=device,
            )
        )
        self.objective = objective

        self.mppi_costs = [
            RunningCost(
                objective,
                idx=i,
                projector=prober if projected_cost else None,
            )
            for i in range(n_envs)
        ]

        if isinstance(model.spatial_repr_dim, int):
            nx = torch.Size((model.spatial_repr_dim,))
        else:
            nx = torch.Size(model.spatial_repr_dim)

        self.ctrls = [
            MPPI(
                self.dynamics,
                running_cost=self.mppi_costs[i],
                nx=nx,
                noise_sigma=noise_sigma,
                num_samples=config.num_samples,
                lambda_=config.lambda_,
                device=device,
                action_normalizer=action_normalizer,
                u_per_command=-1,
                latent_actions=latent_actions,
                z_reg_coeff=config.z_reg_coeff,
                step_dependent_dynamics=self.dynamics.use_l2_conditioning,
            )
            for i in range(n_envs)
        ]
        self.last_plan_size = None
        self.num_refinement_steps = num_refinement_steps
        self.l2 = l2

    @torch.no_grad()
    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        repr_input: bool = True,
        curr_propio_pos: Optional[torch.Tensor] = None,
        curr_propio_vel: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        """_summary_
        Args:
            current_state (bs, ch, w, h): representation of current obs
            plan_size (int): how many predictions to make into the future

        Returns:
            predictions (plan_size + 1, bs, n)
            actions (bs, plan_size, 2)
            locations (plan_size + 1, bs, 2) - probed locations from the predictions
            losses - set to None for now
        """
        batch_size = current_state.shape[0]
        self.dynamics.before_planning_callback()

        raw_state = None
        if not repr_input:
            raw_state = current_state
            if self.model.backbone.config.propio_dim:
                if curr_propio_vel is not None and curr_propio_pos is not None:
                    curr_propio_states = torch.cat(
                        [curr_propio_pos, curr_propio_vel], dim=-1
                    )
                elif curr_propio_vel is not None:
                    curr_propio_states = curr_propio_vel
                elif curr_propio_pos is not None:
                    curr_propio_states = curr_propio_pos
                else:
                    raise ValueError("Need proprio states to plan")

                backbone_output = self.model.backbone(
                    current_state.cuda(), propio=curr_propio_states.cuda()
                )
            else:
                backbone_output = self.model.backbone(current_state.cuda())

            current_state = backbone_output.encodings

        actions = []
        for i in range(batch_size):
            if self.last_plan_size is not None and plan_size < self.last_plan_size:
                for _ in range(self.last_plan_size - plan_size):
                    self.ctrls[i].shift_nominal_trajectory()

            self.ctrls[i].change_horizon(plan_size)

            # add refinement steps?
            if self.dynamics.use_l2_conditioning:
                raw_obs_i = raw_state[i] if raw_state is not None else None
                propio_pos_i = (
                    curr_propio_pos[i] if curr_propio_pos is not None else None
                )
                propio_vel_i = (
                    curr_propio_vel[i] if curr_propio_vel is not None else None
                )
                self.dynamics.set_rollout_context(
                    raw_obs=raw_obs_i,
                    propio_pos=propio_pos_i,
                    propio_vel=propio_vel_i,
                )
            actions.append(
                self.ctrls[i].command(
                    current_state[i],
                    shift_nominal_trajectory=False,
                )
            )

        actions = torch.stack(actions)

        if self.dynamics.use_l2_conditioning and raw_state is not None:
            conditioned_actions = []
            for i in range(batch_size):
                cond = self.dynamics._compute_conditioned_actions(
                    actions=actions[i],
                    raw_obs=raw_state[i],
                    propio_pos=(
                        curr_propio_pos[i] if curr_propio_pos is not None else None
                    ),
                    propio_vel=(
                        curr_propio_vel[i] if curr_propio_vel is not None else None
                    ),
                )
                if cond is None:
                    conditioned_actions.append(actions[i])
                else:
                    if cond.dim() == 3:
                        cond = cond[:, 0]
                    conditioned_actions.append(cond.to(actions.device))
            actions = torch.stack(conditioned_actions)

        pred_encs, pred_obs = self.dynamics(
            state=current_state,
            action=actions.permute(1, 0, 2),
            only_return_last=False,
            flatten_output=False,
        )

        if self.action_normalizer is not None:
            actions = self.action_normalizer(actions)

        actions = self.normalizer.unnormalize_action(actions)

        self.dynamics.after_planning_callback()
        self.last_plan_size = plan_size

        losses = [0]

        if self.prober is not None:
            pred_locs = torch.stack([self.prober(x) for x in pred_obs])
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_locs
            ).detach()
        else:
            unnormed_locations = None

        return PlanningResult(
            pred_encs=pred_encs,
            pred_obs=pred_obs,
            actions=actions,
            locations=unnormed_locations,
            losses=losses,
        )

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.objective.set_target(targets, repr_input=repr_input)
