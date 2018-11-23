"""PPO agent
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent:
    buffer_attrs = [
        "states", "actions", "next_states",
        "rewards", "log_probs", "values", "dones",
    ]

    def __init__(self, env, model, rollout=4, tmax=50, n_epoch=20,
                 batch_size=128, gamma=0.995, delta=0.96, eps=0.20, device="cpu"):
        """PPO Agent
        Parameters
        ----------
        environment :
            対象環境オブジェクト
        network :
            予測モデルそのもの
        """
        self.env = env
        self.model = model
        self.opt_actor = optim.Adam(model.fc_actor.parameters(), lr=1e-4)
        self.opt_critic = optim.Adam(model.fc_critic.parameters(), lr=1e-4)
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.tmax = tmax
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
        self.device = device

        self.reset()

    def to_tensor(self, x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(self.device)

    def reset(self):
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.last_states = self.to_tensor(env_info.vector_observations)

    def collect_trajectories(self):
        buffer = dict([(k, []) for k in self.buffer_attrs])

        for t in range(self.tmax):
            memory = {}

            # draw action from model
            memory["states"] = self.last_states
            with torch.no_grad():
                pred = self.model(memory["states"])
            memory["actions"], memory["log_probs"], _, memory["values"] = pred

            # one step forward
            env_info = self.env.step(memory["actions"].numpy())[self.brain_name]
            memory["next_states"] = self.to_tensor(env_info.vector_observations)
            memory["rewards"] = self.to_tensor(env_info.rewards)
            memory["dones"] = self.to_tensor(env_info.local_done, dtype=np.uint8)

            # stack one step memory to buffer
            for k, v in memory.items():
                buffer[k].append(v.unsqueeze(0))

            self.last_states = memory["next_states"]
            if memory["dones"].any():
                break

        for k, v in buffer.items():
            buffer[k] = torch.cat(v, dim=0)

        return buffer

    def calc_returns(self, rewards, values, last_values):
        n_step, n_agent = rewards.shape

        # Create empty buffer
        GAE = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Set start values
        GAE_current = torch.zeros(n_agent)
        returns_current = last_values
        values_next = last_values

        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]

            # Calculate TD Error
            td_error = rewards_current + self.gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + self.gamma * self.delta * GAE_current
            returns_current = rewards_current + self.gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns

    def step(self):
        """1ステップ進める
        """
        self.model.eval()

        # Collect Trajetories
        trajectories = self.collect_trajectories()

        # Calculate Score (averaged over agents)
        score = trajectories["rewards"].sum(dim=0).mean()

        # Append Values collesponding to last states
        with torch.no_grad():
            last_values = self.model.state_values(self.last_states)
        advantages, returns = self.calc_returns(trajectories["rewards"],
                                                trajectories["values"],
                                                last_values)

        # Mini-batch update
        self.model.train()
        n_sample = advantages.shape[0]
        n_batch = (n_sample - 1) // self.batch_size + 1
        idx = np.arange(n_sample)
        np.random.shuffle(idx)
        for k, v in trajectories.items():
            trajectories[k] = v[idx]
        advantages, returns = advantages[idx], returns[idx]

        for i_epoch in range(self.n_epoch):
            for i_batch in range(n_batch):
                idx_start = self.batch_size * i_batch
                idx_end = self.batch_size * (i_batch + 1)
                (states, actions, next_states, rewards, old_log_probs,
                 old_values, dones) = [trajectories[k][idx_start:idx_end]
                                       for k in self.buffer_attrs]
                advantages_batch = advantages[idx_start:idx_end]
                returns_batch = returns[idx_start:idx_end]

                _, log_probs, entropy, values = self.model(states, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                ratio_clamped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                ratio_PPO = torch.where(ratio < ratio_clamped, ratio, ratio_clamped)
                loss_actor = -torch.mean(ratio_PPO * advantages_batch) - 0.01 * torch.mean(entropy)
                loss_critic = (returns_batch - values).pow(2).mean()

                self.opt_actor.zero_grad()
                loss_actor.backward()
                self.opt_actor.step()
                del(loss_actor)

                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()
                del(loss_critic)

        return score
