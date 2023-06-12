import math
import random
from copy import deepcopy
import os.path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from .model import Actor, Critic
from .env import Env, EnvBatch


class Trainer:
    def __init__(self, batch_size, num_episode, save_path, device="cuda"):
        self.batch_size = batch_size

        self.device = device
        self.save_path = save_path

        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.num_episode = num_episode

        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=0.0001)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=0.001)

        self.test_env = self.get_test()

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])

    def save(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
            },
            os.path.join(self.save_path, "model.pt"),
        )

    def render(self, episode):
        save_path = os.path.join(self.save_path, f"episode_{episode}.png")
        test_env = deepcopy(self.test_env)

        orients = []

        with torch.no_grad():
            done = False
            i = 0
            while not done and i < 100:  # 15 seconds timeout
                orient, depth_map = test_env.orient, test_env.depth_map
                orients.append(orient.cpu().numpy().copy())

                mean, std = self.actor(orient, depth_map)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

                _, _, done = test_env.step(action)

                i += 1

        orients = np.array(orients)

        _, ax = plt.subplots()
        for i in range(test_env.n_agents):
            ax.plot(orients[:, i, 0], orients[:, i, 1])
            ax.plot(orients[0, i, 0], orients[0, i, 1], marker="o", color="green")
            ax.plot(orients[-1, i, 0], orients[-1, i, 1], marker="x", color="red")

        ax.plot(
            test_env.target[:, 0],
            test_env.target[:, 1],
            marker="*",
            linestyle="None",
            color="blue",
        )

        ax.set(xlabel="X-axis", ylabel="Y-axis", title="Trajectories")
        plt.savefig(save_path)

    def get_test(self):
        env = Env(6, 3)
        for i in range(6):
            env.orient[i, 0] = 5 + 3 * math.cos(i * math.pi / 3)
            env.orient[i, 1] = 5 + 3 * math.sin(i * math.pi / 3)
            env.orient[i, 2] = 0
            env.target[i, 0] = 5 - 2 * math.cos(i * math.pi / 3)
            env.target[i, 1] = 5 - 2 * math.sin(i * math.pi / 3)

        env.orient[:, 2] = (
            (env.orient[:, 1] - 10 / 2).atan2(env.orient[:, 0] - 10 / 2) + math.pi
        ) % (2 * math.pi)

        for i in range(3):
            env.obstacles[i, 0] = 5 + math.cos(i * 2 * math.pi / 3)
            env.obstacles[i, 1] = 5 + math.sin(i * 2 * math.pi / 3)

        return env.to(self.device)

    def get_rand_batch(self):
        env_batch = EnvBatch(self.batch_size, (1, 5), (0, 3))
        env_batch.to(self.device)
        done = [False] * self.batch_size
        return env_batch, done

    def train(self):
        self.actor.train()
        self.critic.train()

        for episode in tqdm(range(self.num_episode)):
            orients = []
            depth_maps = []
            actions = []
            log_probs = []

            # with torch.no_grad():
            env_batch, done = self.get_rand_batch()
            i = 0
            while not all(done) and i < 100:  # 10 seconds timeout
                orient, depth_map = env_batch.get_state(done)
                orients.append(orient.detach())
                depth_maps.append(depth_map.detach())

                mean, std = self.actor(orient, depth_map)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

                env_batch.step(action, done)
                actions.append(action)
                log_probs.append(dist.log_prob(action).sum(dim=1))

                i += 1

            batch_r = env_batch.get_reward()
            rewards = torch.tensor(batch_r).to(self.device)

            orients = torch.cat(orients, dim=0)
            depth_maps = torch.cat(depth_maps, dim=0)
            actions = torch.cat(actions, dim=0)
            log_probs = torch.cat(log_probs, dim=0)

            values = self.critic(orients, depth_maps, actions)
            advantages = (rewards - values).detach()

            # update actor
            actor_loss = -(log_probs * advantages).sum()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update critic
            critic_loss = torch.nn.functional.smooth_l1_loss(rewards, values)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            if episode % 100 == 0:
                print(
                    f"Episode: {episode}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}"
                )
                self.render(episode)
