import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, memory, batch_size, seed, lr, gamma, tau, update_network_steps, ddqn=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            memory (ReplayBuffer): The replay buffer for storing xperiences
            batch_size (int): Number of experiences to sample from the memory
            seed (int): The random seed
            lr (float): The learning rate 
            gamma (float): The reward discount factor
            tau (float): For soft update of target parameters
            update_network_steps (int): How often to update the network
            ddqn (bool): Use double DQN network update strategy (default is true)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.update_network_steps = update_network_steps
        self.ddqn = ddqn

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_network_steps
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, internal_state = experiences

        # q_target
        q_target = self.ddqn_q_target(experiences, gamma) if self.ddqn else self.dqn_q_target(experiences, gamma)

        # q
        q = self.qnetwork_local(states).gather(1, actions)

        # loss
        self.optimizer.zero_grad()
        #loss = F.mse_loss(q_target, q)
        td_error = q_target - q
        loss = self.weighted_loss(td_error, weights)
        loss.backward()
        self.optimizer.step()
        
        # update memory 
        self.memory.update(internal_state, td_error.detach())

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def ddqn_q_target(self, experiences, gamma):
        states, actions, rewards, next_states, dones, _, _ = experiences
        q_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        q_next_values = self.qnetwork_target(next_states).gather(1, q_actions)
        q_target = rewards + (gamma * q_next_values * (1 - dones))
        return q_target
        
    def dqn_q_target(self, experiences, gamma):
        print("DQN====")
        states, actions, rewards, next_states, dones, _, _ = experiences
        q_next_values = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + (gamma * q_next_values * (1 - dones))
        return q_target

    def weighted_loss(self, values, weights):
        loss = values ** 2
        loss = weights * loss 
        loss = loss.sum()
        return loss
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
