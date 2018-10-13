from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import os
import argparse
import matplotlib.pyplot as plt

from memory import ReplayBuffer, PrioritizedReplayBuffer
from agent import Agent

# Arguments Parsing Settings
parser = argparse.ArgumentParser()

parser.add_argument('--seed', help="Seed for random number generation", default=0)
parser.add_argument('--checkpoint', help="The model checkpoint file name", default="checkpoint.pth")
parser.add_argument('--reward_plot', help="The reward plot file name", default="reward_plot.png")

# training/testing flags
parser.add_argument('--train', help="train or test (flag)", action="store_true")
parser.add_argument('--test_episodes', help="The number of episodes for testing", default=3)
parser.add_argument('--train_episodes', help="The number of episodes for training", default=500)
parser.add_argument('--eps_start', help="Epsilon start value for exploration/exploitation", default=1.0)
parser.add_argument('--eps_decay', help="Epsilon decay value for exploration/exploitation", default=0.995)
parser.add_argument('--eps_end', help="Epsilon minimum value for exploration/exploitation", default=0.01)
parser.add_argument('--gamma', help="The reward discount factor", default=0.99)
parser.add_argument('--tau', help="For soft update of target parameters", default=1e-3)
parser.add_argument('--lr', help="The learning rate ", default=5e-4)
parser.add_argument('--baseline', help="Baseline for updating network weights during training", default=float(0.00025/4.0))
parser.add_argument('--update_network_steps', help="How often to update the network", default=4)

# replay memory 
parser.add_argument('--prioritized_buffer', help="Use prioritized replay buffer (flag)", action="store_true")
parser.add_argument('--buffer_size', help="The replay buffer size", default=int(1e5))
parser.add_argument('--batch_size', help="The mini batch size", default=64)
parser.add_argument('--update_buffer_steps', help="How often to update the buffer", default=10000)
parser.add_argument('--alpha', help="The priority exponent", default=0.7)
parser.add_argument('--beta', help="The importance sampling exponent", default=0.5)
parser.add_argument('--beta_inc', help="The importance sampling exponent increment", default=0.00001)


def create_environment():
    env = UnityEnvironment("Banana_Windows_x86_64/Banana.app")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    
    return env, brain, brain_name, action_size, state_size

def test(agent, env, brain, brain_name, n_episodes):
    if os.path.isfile(args.checkpoint):
        print("loading checkpoint for agent ...")
        agent.qnetwork_local.load_state_dict(torch.load(args.checkpoint))

    # watch an untrained agent
    score = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        while True:
            action = agent.act(state)
            
            env_info = env.step(int(action))[brain_name]        # send the action to the environment
            state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            score += reward
            if done:
                break 
            
    print("score:", float(score)/float(n_episodes))

def train(agent, env, brain, brain_name, n_episodes, eps_start, eps_end, eps_decay):
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps)
            
            env_info = env.step(int(action))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break 

        scores_window.append(score)       # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), args.checkpoint)
            plot_rewards(scores_window)
            break

def plot_rewards(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(args.reward_plot, transparent=True)
    plt.close()
            
if __name__ == '__main__':
    args = parser.parse_args()
    
    # environment
    env, brain, brain_name, action_size, state_size = create_environment()

    # replay memory
    if args.prioritized_buffer:
        memory = PrioritizedReplayBuffer(action_size=action_size, 
                                         state_size=state_size, 
                                         buffer_size=args.buffer_size,
                                         batch_size=args.batch_size,
                                         update_buffer_steps=args.update_buffer_steps,
                                         seed=args.seed,
                                         alpha=args.alpha,
                                         beta=args.beta,
                                         beta_inc=args.beta_inc)
    else:
        memory = ReplayBuffer(action_size=action_size, 
                                         state_size=state_size, 
                                         buffer_size=args.buffer_size,
                                         batch_size=args.batch_size,
                                         seed=args.seed)

    # agent
    agent = Agent(state_size=state_size, 
                  action_size=action_size, 
                  batch_size=args.batch_size,
                  seed=args.seed,
                  memory=memory,
                  lr=args.lr,
                  gamma=args.gamma,
                  tau=args.tau,
                  baseline=args.baseline,
                  update_network_steps=args.update_network_steps)

    if args.train:
        train(agent, env, brain, brain_name, 
              n_episodes=args.train_episodes, 
              eps_start=args.eps_start, 
              eps_end=args.eps_end, 
              eps_decay=args.eps_decay)
    else:
        test(agent, env, brain, brain_name, n_episodes=args.test_episodes)

    env.close()
