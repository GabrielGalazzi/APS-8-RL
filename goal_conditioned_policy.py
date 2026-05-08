from typing import Any

import torch as th
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy


class GoalConditionedMlpExtractor(nn.Module):
    """
    Actor/critic MLP extractor with a small latent goal head on the actor path.

    The actor receives [features, tanh(goal_head(features))], while the critic
    receives the unchanged features.
    """

    def __init__(
        self,
        feature_dim: int,
        goal_dim: int,
        net_arch: list[int] | dict[str, list[int]] | None,
        activation_fn: type[nn.Module],
        device: th.device,
    ) -> None:
        super().__init__()
        net_arch = [] if net_arch is None else net_arch

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        self.goal_head = nn.Linear(feature_dim, goal_dim)

        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim + goal_dim
        last_layer_dim_vf = feature_dim + goal_dim 

        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)
        self.to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        goal = th.tanh(self.goal_head(features))
        action_input = th.cat((features, goal), dim=1)
        return self.policy_net(action_input)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        goal = th.tanh(self.goal_head(features))
        critic_input = th.cat((features, goal), dim=1)
        return self.value_net(critic_input)


class GoalConditionedMultiInputLstmPolicy(RecurrentMultiInputActorCriticPolicy):
    def __init__(self, *args: Any, goal_dim: int = 16, **kwargs: Any) -> None:
        self.goal_dim = goal_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GoalConditionedMlpExtractor(
            self.lstm_output_dim,
            goal_dim=self.goal_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(goal_dim=self.goal_dim)
        return data
