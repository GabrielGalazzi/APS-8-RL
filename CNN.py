import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CPPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Read dimensions from the actual env spaces so the extractor stays in
        # sync if observation fields change later.
        agent_dim = observation_space["agent"].shape[0]
        neighbors_shape = observation_space["neighbors"].shape

        # Process the local 3x3 map as a channel-first image:
        # channel 0 = obstacle/wall, channel 1 = visited, channel 2 = unvisited.
        # The wider second conv gives larger grids more local geometry capacity.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            # Compute the flattened CNN size dynamically instead of hard-coding
            # dimensions from a particular grid or convolution setup.
            sample = th.zeros(1, *neighbors_shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        # Small MLP for scalar global state: normalized agent coordinates and
        # current coverage ratio.
        self.agent_mlp = nn.Sequential(
            nn.Linear(agent_dim, 32),
            nn.ReLU(),
        )

        # Fuse local geometry and scalar progress. RecurrentPPO's LSTM handles
        # temporal state outside the feature extractor.
        fused_dim = cnn_out_dim + 32
        self.combined = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ELU(),
        )

    def forward(self, observations):
        # Dict observations arrive batched from SB3. Neighbor tensors are already
        # channel-first, so no unsqueeze/reshape is needed before the CNN.
        agent = observations["agent"]
        nbrs = observations["neighbors"]

        cnn_out = self.cnn(nbrs)
        agent_out = self.agent_mlp(agent)

        combined = th.cat((cnn_out, agent_out), dim=1)
        return self.combined(combined)
