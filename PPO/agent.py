"""PPO agent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent:
    buffer_attrs = [
        "states", "actions", "next_states",
        "rewards", "log_probs", "values", "done",
    ]

    def __init__(self, env, model, rollout=4, tmax=50, n_epoch=20,
                 gamma=0.995, delta=0.3, eps=0.1, device="cpu"):
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
        self.opt_actor = optim.Adam(model.fc_actor.paramters(), lr=1e-4)
        self.opt_critic = optim.Adam(model.fc_critic.parameters(), lr=1e-4)
        self.n_agent = n_agent
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.tmax = tmax
        self.n_epoch = n_epoch
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
        self.device = device

        self.reset()

    def to_tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def reset(self):
        env_info = env.reset(train_mode=False)[brain_name]
        self.last_states = self.to_tensor(env_info.vector_observations)

    def collect_trajectories(self):
        buffer = dict([(k, []) for k in self.buffer_attrs])

        for t in range(self.tmax):
            memory = {}

            # draw action from model
            memory["states"] = self.last_states
            pred = self.model(last_states)
            pred_detached = [v.detach() for v in pred]
            memory["actions"], memory["log_probs"], memory["values"] = pred_detached

            # one step forward
            env_info = env.step(actions)[brain_name]
            memory["next_states"] = self.to_tensor(env_info.vector_observations)
            memory["rewards"] = self.to_tensor(env_info.rewards)
            memory["dones"] = self.to_tensor(env_info.local_done)

            # stack one step memory to buffer
            for k, v in memory.items():
                buffer[k].append(v)

            self.last_states = memory["next_states"]
            if np.any(memory["dones"]):  # TODO : dones are boolean, but torch.tesor
                break

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
        # Collect Trajetories
        trajectories = self.collect_trajectories()

        # Append Values collesponding to last states
        last_values = self.model.state_values(self.last_states)
        advantages, returns = calc_returns(trajectories["rewards"],
                                           trajectories["values"], last_values)

        (states, actions, next_states, rewards, old_log_probs,
         old_values, done) = [trajectories[k] for k in buffer_attrs]

        # Mini-batch update
        for epoch in range(self.n_epoch):
            log_probs, _, values = model.forward(states, actions)

            # Actor Update
            ratio = torch.exp(log_probs - old_log_probs + 1e-6)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            ratio_PPO = torch.where(ratio < ratio_clamped, ratio, ratio_clamped)
            loss_actor = torch.mean(ratio_PPO * advantages)

            self.opt_actor.zero_grad()
            loss_actor.backward()
            self.opt_actor.step()
            del(loss_actor)

            # Critic Update
            loss_critic = (returns - values).pow(2).mean()
            self.opt_critic.zero_grad()
            loss_critic.backward()
            self.opt_critic.step()
            del(loss_critic)
