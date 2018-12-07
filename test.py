import sys
import numpy as np
import torch
from unityagents import UnityEnvironment

from PPO import GaussianActorCriticNetwork
from PPO import PPOAgent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if sys.platform == "darwin":
    binary_path = "./bin/Reacher.app"
elif sys.platform == "linux":
    binary_path = "./bin/Reacher_Linux_NoVis/Reacher.x86_64"
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




def train_test():
    print(" --- initialize env   ... ", end=" ")
    env = UnityEnvironment(file_name=binary_path)
    print("Done.")
    n_agent, state_dim, action_dim = get_env_info(env)
    print(" --- initialize model ... ", end=" ")
    model = GaussianActorCriticNetwork(state_dim, action_dim, hiddens=[512, 256])
    model = model.to(device)
    print("Done.")
    print(" --- initialize agent ... ", end=" ")
    agent = PPOAgent(env, model, tmax=100, n_epoch=10, batch_size=128, eps=0.1, device=device)
    print("Done.")
    n_step = 20000
    n_episodes = 0
    for step in range(n_step):
        agent.step()
        scores = agent.scores_by_episode
        if n_episodes < len(scores):
            n_episodes = len(scores)
            print(f" episode {n_episodes} : rewards = {scores[-1]:.2f}", end="")
            if 100 <= n_episodes:
                rewards_ma = np.mean(scores[-100:])
                print(f", last 100 mean = {rewards_ma:.2f}")
            else:
                print()

        sys.stdout.flush()

if __name__ == "__main__":
    train_test()
