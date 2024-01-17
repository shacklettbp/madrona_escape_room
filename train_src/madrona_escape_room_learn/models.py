import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 2, 1)

        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = inputs.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(F.relu(self.conv4(x)))

        # print("CNN forward output shape: ", x.shape)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_channels, num_layers):
        super().__init__()

        layers = [
            nn.Linear(input_dim, num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.LayerNorm(num_channels))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, inputs):
        # convert inputs to half precision
        # print("MLP forward output shape: ", self.net(inputs).shape)
        inputs = inputs.half()
        return self.net(inputs)

class LinearLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)

class DenseLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, dtype):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, total_action_dim),
        )

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl[0].weight, gain=0.01)
        nn.init.constant_(self.impl[0].bias, 0)
        nn.init.orthogonal_(self.impl[3].weight, gain=0.01)
        nn.init.constant_(self.impl[3].bias, 0)

class DenseLayerCritic(Critic):
    def __init__(self, dtype):
        super().__init__(nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ))

        nn.init.orthogonal_(self.impl[0].weight)
        nn.init.constant_(self.impl[0].bias, 0)
        nn.init.orthogonal_(self.impl[3].weight)
        nn.init.constant_(self.impl[3].bias, 0)

