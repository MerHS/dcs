import random
import math
from itertools import zip_longest

import numpy as np
import torch

TIME_STEP = 0.1
W_GOAL = 4.0
W_COLLISION = -3.0
W_VELOCITY = -4.0
W_THETA = -1.0
RADIUS = 0.2
PI_20 = math.pi / 20
GAMMA = 0.95

WALL_H = 10
WALL_W = 10


def rel_angle(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def flood(x, x_min, x_max):
    return abs(min(x - x_min, 0)) + abs(max(x - x_max, 0))


class Env:
    def __init__(self, n_agents, n_obstacles):
        self.n_agents = n_agents
        self.orient = torch.zeros(n_agents, 3)  # x, y, theta
        self.target = torch.zeros(n_agents, 2)  # x, y
        self.obstacles = torch.zeros(n_obstacles, 2)  # x, y
        self.depth_map = torch.zeros(n_agents, 4, 21)
        self.next_map = torch.zeros(n_agents, 21)

    @staticmethod
    def gen_rand(n_agents, n_obstacles):
        env = Env(n_agents, n_obstacles)
        x = torch.rand(n_agents) * ((WALL_W - 2) / 2) + 1
        y = torch.rand(n_agents) * ((WALL_H - 2) / 2) + 1
        env.orient[:, 0] = x
        env.orient[:, 1] = torch.rand(n_agents) * ((WALL_H - 2) / 2) + 1
        env.orient[:, 2] = ((x - WALL_H / 2).atan2(y - WALL_W / 2) + math.pi) % (
            2 * math.pi
        )
        env.target[:, 0] = WALL_W - x + (2 * torch.rand(n_agents) - 1)
        env.target[:, 1] = WALL_H - y + (2 * torch.rand(n_agents) - 1)

        env.obstacles[:, 0] = torch.rand(n_obstacles) * (WALL_W - 4) + 2
        env.obstacles[:, 0] = torch.rand(n_obstacles) * (WALL_H - 4) + 2

        env.init_depth_map()

        return env

    def init_depth_map(self):
        for _ in range(4):
            self.step(torch.zeros(self.n_agents, 3))

    def _init_next_map(self):
        for i in range(self.n_agents):
            for ray in range(21):
                ray_angle = (ray - 10) * PI_20
                angle = (self.orient[i, 2].item() + ray_angle) % (2 * math.pi)
                x, y = self.orient[i, :2]

                depth_up = (WALL_H - y) / (math.sin(angle) + 1e-8)
                depth_down = y / (math.sin(angle - math.pi) + 1e-8)
                depth_left = x / (math.cos(angle - math.pi) + 1e-8)
                depth_right = (WALL_W - x) / (math.cos(angle) + 1e-8)
                depth_ud = depth_up if depth_up >= 0 else depth_down
                depth_lr = depth_left if depth_left >= 0 else depth_right
                depth = min(depth_ud, depth_lr)

                self.next_map[i, ray] = depth

    def _update_next_map(self, i, target_vec, i_vec, i_angle):
        vector = target_vec - i_vec
        dist = vector.norm().item()
        angle = rel_angle(i_angle.item(), math.atan2(vector[1], vector[0]))
        view_angle = math.atan2(RADIUS, dist)

        for ray in range(21):
            ray_angle = (ray - 10) * PI_20
            if abs(rel_angle(angle, ray_angle)) < view_angle:
                self.next_map[i, ray] = min(self.next_map[i, ray], dist)
        return dist

    def step(self, action):
        # action: (n_agents, 3)
        rewards = np.array([0.0 for _ in range(self.n_agents)])

        orig_dist = (self.target - self.orient[:, :2]).norm(dim=1)

        # update orientation
        self.orient[:, :2] += action[:, :2] * TIME_STEP

        # bound orientation
        self.orient[:, :2][self.orient[:, :2] < 0] = 0
        self.orient[:, 0][self.orient[:, 0] > WALL_W] = WALL_W
        self.orient[:, 1][self.orient[:, 1] > WALL_H] = WALL_H

        self.orient[:, 2] = (self.orient[:, 2] + action[:, 2] * TIME_STEP) % (
            2 * math.pi
        )

        old_depth_map = self.depth_map
        self.depth_map = old_depth_map.clone()

        self.depth_map[:, :3, :] = old_depth_map[:, 1:, :]
        self._init_next_map()

        for i in range(self.n_agents):
            collision_reward = 0

            # update depth map
            for j in range(self.n_agents):
                if i == j:
                    continue

                dist = self._update_next_map(
                    i, self.orient[j, :2], self.orient[i, :2], self.orient[i, 2]
                )

                if dist < 2 * RADIUS:
                    collision_reward = W_COLLISION / 2

            for j in range(self.obstacles.shape[0]):
                dist = self._update_next_map(
                    i, self.obstacles[j], self.orient[i, :2], self.orient[i, 2]
                )

                if dist < 2 * RADIUS:
                    collision_reward = W_COLLISION

            rewards[i] += collision_reward

            # r_smooth
            rewards[i] += W_VELOCITY * flood(action[i, :2].norm(), -0.5, 1.5)
            rewards[i] += W_THETA * flood(action[i, 2], -math.pi / 4, math.pi / 4)

        self.depth_map[:, 3, :] = self.next_map

        next_dist = (self.target - self.orient[:, :2]).norm(dim=1)
        done = (next_dist < RADIUS).all().item()

        reward_goal = W_GOAL * (orig_dist - next_dist)

        rewards += reward_goal.cpu().numpy()

        return (self.orient, self.depth_map), rewards, done

    def to(self, device):
        self.orient = self.orient.to(device)
        self.target = self.target.to(device)
        self.obstacles = self.obstacles.to(device)
        self.depth_map = self.depth_map.to(device)
        self.next_map = self.next_map.to(device)

        return self


class EnvBatch:
    def __init__(self, n_batch, range_agents, range_obstacles):
        self.n_batch = n_batch
        self.envs = [
            Env.gen_rand(
                random.randint(*range_agents),
                random.randint(*range_obstacles),
            )
            for _ in range(n_batch)
        ]

        self.rewards = [[] for _ in range(n_batch)]

    def step(self, action, done):
        # action: (undone n_batch * n_agents, 3)
        # done: (n_batch)

        n_agents = 0
        for i in range(self.n_batch):
            if done[i]:
                continue

            this_agents = self.envs[i].n_agents
            _, rewards, next_done = self.envs[i].step(
                action[n_agents : n_agents + this_agents]
            )

            self.rewards[i].append(rewards)

            done[i] = next_done
            n_agents += this_agents

    def get_state(self, done):
        orient = torch.cat(
            [env.orient for i, env in enumerate(self.envs) if not done[i]], dim=0
        )
        depth_map = torch.cat(
            [env.depth_map for i, env in enumerate(self.envs) if not done[i]], dim=0
        )

        return orient, depth_map

    def get_reward(self):
        cumul_returns = []
        for rewards in self.rewards:
            ret = []
            R = 0
            for r in rewards[::-1]:
                R = r + GAMMA * R
                ret.insert(0, R)
            cumul_returns.append(ret)

        # transpose list of lists
        ret_vals = []
        for rewards in zip_longest(*cumul_returns, fillvalue=None):
            ret_vals.extend(r for r in rewards if r is not None)

        return np.concatenate(ret_vals)

    def to(self, device):
        for env in self.envs:
            env.to(device)

        return self
