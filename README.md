# Deep Queue Network Reinforcement Learning Agent

This repository contains a DQN agent implemented in PyTorch to solve a navigation task. The environment is defined in the Unity engine, and the communication is based on the Unity ML Agents API.

See the Report.md file for in depth details about the algorithms used, and the organization of the source code.

## Goal

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

The source code is implemented in Python, and uses PyTorch as the Machine Learning framework. It should run on any compatible operating system. However, the instructions below are for Windows only. 

1. [Install Anaconda for Windows](https://conda.io/docs/user-guide/install/windows.html)
    * Select the Python 3x version
2. [Install PyTorch via Anaconda](https://pytorch.org/get-started/locally/)
3. Download the [Unity environment](https://drive.google.com/open?id=1Pjl54zFSBf2DreF3jfNLvHnkBm3VNEkJ)
    * Place the file in the repository folder, and unzip (or decompress) the file.
4. Download the [model checkpoint](https://drive.google.com/open?id=1Le5DI8kVOhiUJhyAYar9jU7ArpSWfD3t)
    * Place the file in the repository folder

## Instructions

The main.py is the application entry point. To show all available options:

```bash
python main.py --help
```

To train the agent:

```bash
python main.py --train
```

Note: By default, the agent uses the Double DQN algorithm with a uniform sampling replay buffer. The training runs 500 episodes, and saves the model checkpoint in the current directory if the goal is achieved.

To test the agent using a model checkpoint:

```bash
python main.py
```

By default the agent uses the Double DQN strategy, and a uniform sampling replay buffer. You can select the original DQN algorithm as follows:

```bash
python main.py --train --dqn
```

Instead of the original uniform sampling buffer, you can configure the agent to use a rank-based prioritized replay buffer as follows (experimental):

```bash
python main.py --train --prioritized_buffer
```

In addition, many hyper parameters can be customized such as the learning rate, the reward discount factor gamma, the epslon greedy policy exploration/exploitation, number of training and testing episodes. Check the --help for all available options.
