import torch

import torch.nn as nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.internal = nn.Sequential(
            nn.Linear(3, 32),
            # nn.BatchNorm1d(32)
            nn.ELU(),
            nn.Linear(32, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
        )

        self.ext1 = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=0),
            nn.ELU(),
        )

        self.ext2 = nn.Sequential(
            nn.Linear(38, 32),
            nn.ELU(),
        )

        self.merge_layer = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
        )

        self.mean = nn.Sequential(
            nn.Linear(32, 3),
            nn.Tanh(),
        )

        self.std = nn.Linear(32, 3)

    def forward(self, orient, depth_map):
        # orient: (B * n_agents, 3)
        # depth_map: (B * n_agents, 4, 21)
        orient = orient.reshape(-1, 3)
        depth_map = depth_map.reshape(-1, 1, 4, 21)
        internal = self.internal(orient)
        ext1 = self.ext1(depth_map)
        ext2 = self.ext2(ext1.reshape(-1, 38))

        merged = self.merge_layer(torch.cat([internal, ext2], dim=1))
        mean = self.mean(merged) * 2.5  # velocity range: [-2.5, 2.5]
        std = torch.exp(self.std(merged)) + 1e-5

        return mean.reshape(-1, 3), std.reshape(-1, 3)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.internal = nn.Sequential(
            nn.Linear(3, 32),
            # nn.BatchNorm1d(32)
            nn.ELU(),
            nn.Linear(32, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
        )

        self.ext1 = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=0),
            nn.ELU(),
        )

        self.ext2 = nn.Sequential(
            nn.Linear(38, 32),
            nn.ELU(),
        )

        self.action = nn.Sequential(
            nn.Linear(64 + 32 + 3, 32), nn.ELU(), nn.Linear(32, 1)
        )

    def forward(self, orient, depth_map, action):
        # orient: (B * n_agents, 3)
        # depth_map: (B * n_agents, 4, 21)
        # action: (B * n_agents, 3)
        orient = orient.reshape(-1, 3)
        depth_map = depth_map.reshape(-1, 1, 4, 21)
        internal = self.internal(orient)
        ext1 = self.ext1(depth_map)
        ext2 = self.ext2(ext1.reshape(-1, 38))

        value = self.action(torch.cat([internal, ext2, action.reshape(-1, 3)], dim=1))

        return value.reshape(-1)
