"""PPO agent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOAgent:
    def __init__(self, env, model, rollout=4, gamma=0.995, delta=0.3):
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
        self.n_agent = n_agent
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.tmax = tmax
        self.reset()

    def reset(self):
        env_info = env.reset(train_mode=False)[brain_name]
        self.states = env_info.vector_observations

    def collect_trjectories(self):
        (buf_states, buf_actions, buf_next_states,
         buf_rewards, buf_log_probs, buf_values) = [], [], [], [], [], []

        states = self.states
        for t in range(tmax):
            # draw action from model
            actions, log_probs, values = self.model(self.states)

            # one step forward
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # memorize to buffer
            buf_states.append(self.states)
            buf_actions.append(actions)
            buf_next_states.append(next_states)
            buf_rewards.append(rewards)
            buf_log_probs.append(log_probs)
            buf_values.append(values)

            states = next_states
            if np.any(dones):
                break

        return (buf_states, buf_actions, buf_next_states,
                buf_rewards, buf_log_probs, buf_values)

    def calc_returns(self, rewards, values, last_values):
        GAE = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        GAE_current = np.zeros([1, rewards.shape[1]])
        returns_current = last_values
        values_next = last_values

        for irow in reversed(range(rewards.shape[0])):
            values_current = values[irow]
            rewards_current = rewards[irow]

            td_error = rewards_current + self.gamma * values_next - values_current
            GAE_current = td_error + self.gamma * self.delta * GAE_current
            returns_current = rewards_current + self.gamma * return_current

            GAE[irow] = GAE_current
            returns_current[irow] = returns_current

            values_next = values_current

    def to_tensor(self, np_array):
        return torch(np_array).float().to(self.device)

    def step(self):
        """1ステップ進める
        """
        # Collect Trajetories
        (states, actions, next_states,
         rewards, log_probs, values) = self.collect_trjectories()

        # Append Values collesponding to last states
        _, _, _, _, last_values = self.model(states[-1])
        GAE, returns = calc_returns(rewards, values, last_values)

        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        next_states = self.to_tensor(next_states)
        rewards = self.to_tensor(rewards)
        log_probs = self.to_tensor(log_probs)
        values = self.to_tensor(values)

        for epoch in range(n_epoch):


        # reward の計算 .. 割引ありの reward を算出
        # 割引ありの reward から value を引いて advantage としておく

        # ミニバッチ更新する
        # 綺麗にやるのであれば、shuffle すべきところだが
        # on-policy の順序に準じる
