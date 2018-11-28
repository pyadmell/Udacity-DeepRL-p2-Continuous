"""
"""

import sys
import numpy as np
import torch
from unityagents import UnityEnvironment

from PPO import GaussianActorCriticNetwork
from PPO import PPOAgent


if sys.platform == "darwin":
    binary_path = "./bin/Reacher.app"
else:
    binary_path = "./bin/Reacher_Windows_x86_64/Reacher.exe"


def get_env_info(env):
    # reset the environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    n_agent = len(env_info.agents)
    action_dim = brain.vector_action_space_size
    states = env_info.vector_observations
    state_dim = states.shape[1]

    return n_agent, state_dim, action_dim


def simple_test():
    model = GaussianActorCriticNetwork(state_dim=4, action_dim=4)
    xvals = np.random.randn(100, 4)
    x = torch.from_numpy(xvals).float()
    acts = np.random.randn(100, 4)
    actions = torch.from_numpy(xvals).float()
    print(" ---- x")
    print(f"   shape = {x.shape}")
    print(x[:5])
    print(" ---- actions")
    print(actions[:5])
    print(f"   shape = {actions.shape}")
    actions, log_probs, entropy, v = model.forward(x, actions)
    print(" ---- actions")
    print(actions[:5])
    print(f"   shape = {actions.shape}")
    print(" ---- log_probs")
    print(log_probs[:5])
    print(f"   shape = {log_probs.shape}")
    print(" ---- v")
    print(v[:5])
    print(f"   shape = {v.shape}")


def agent_test():
    print(" --- initialize env   ... ", end=" ")
    env = UnityEnvironment(file_name=binary_path)
    print("Done.")
    n_agent, state_dim, action_dim = get_env_info(env)
    print(" --- initialize model ... ", end=" ")
    model = GaussianActorCriticNetwork(state_dim, action_dim)
    print("Done.")
    print(" --- initialize agent ... ", end=" ")
    agent = PPOAgent(env, model, tmax=1000)
    print("Done.")
    print(agent.last_states)
    print(" --- collect trajectories ... ", end=" ")
    trajectories = agent.collect_trajectories()
    print("Done.")
    for k, v in trajectories.items():
        print(f" {k} : {v.shape}")
    print(" --- calc returnes ... ", end=" ")
    last_values = model.state_values(agent.last_states)
    advantages, returns = agent.calc_returns(trajectories["rewards"],
                                             trajectories["values"], last_values)
    print("Done.")
    print("  advantages -> ", advantages.shape)
    print("  returns -> ", returns.shape)

def train_test():
    print(" --- initialize env   ... ", end=" ")
    env = UnityEnvironment(file_name=binary_path)
    print("Done.")
    n_agent, state_dim, action_dim = get_env_info(env)
    print(" --- initialize model ... ", end=" ")
    model = GaussianActorCriticNetwork(state_dim, action_dim,
        hiddens_actor=[32, 32], hiddens_critic=[32, 32], sigma=0.1)
    print("Done.")
    print(" --- initialize agent ... ", end=" ")
    agent = PPOAgent(env, model, tmax=2048, n_epoch=5, batch_size=128, gamma=0.995)
    print("Done.")
    n_step = 300
    for step in range(n_step):
        agent.reset()
        score = agent.step()
        print(f"{step+1:04d}/{n_step:04d} score = {score:.2f}")
        sys.stdout.flush()

if __name__ == "__main__":
    train_test()
