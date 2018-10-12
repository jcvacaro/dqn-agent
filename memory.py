import math
import heapq
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            state_size (int): dimension of each state
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.ones(self.batch_size, 1, dtype=torch.float, device=device)
  
        return (states, actions, rewards, next_states, dones, weights, None)

    def update(self, entries, errors):
        pass
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size, batch_size, update_buffer_steps, seed, alpha, beta, beta_inc):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            state_size (int): dimension of each state
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.memory_entry_counter = 0
        self.update_buffer_steps = update_buffer_steps
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
    
    def add(self, state, action, reward, next_state, done, error=1.0):
        """Add a new experience to memory."""
        e = experience(state, action, reward, next_state, done)
        entry = [error, self.memory_entry_counter, e]
        if len(self.memory) >= self.buffer_size:
            heapq.heapreplace(self.memory, entry)
        else:
            heapq.heappush(self.memory, entry)
            
        # sort memory
        self.memory_entry_counter = abs(self.memory_entry_counter + 1)
        if (self.memory_entry_counter % self.update_buffer_steps) == 0:
            heapq.heapify(self.memory)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        states = np.zeros((self.batch_size, self.state_size), dtype=np.float)
        actions = np.zeros((self.batch_size,1), dtype=np.int32)
        rewards = np.zeros((self.batch_size,1), dtype=np.float)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float)
        dones = np.zeros((self.batch_size,1), dtype=np.uint8)
        p_j = np.zeros((self.batch_size,1), dtype=np.float)
        entries = []
        
        segment_size = len(self.memory) // self.batch_size
        for i in range(self.batch_size):
            start = i * segment_size
            end = len(self.memory) if i==self.batch_size-1 else (i+1)*segment_size
            j = random.randint(start, end-1)
            entries.append(self.memory[j])
            [error, counter, e] = entries[-1]
            states[i,:] = e.state
            actions[i] = e.action
            rewards[i] = e.reward
            next_states[i,:] = e.next_state
            dones[i] = e.done
            p_j[i] = error

        # place tensors in GPU for faster calculations
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        p_j = torch.from_numpy(p_j).float().to(device)
        
        # the math
        p_j = p_j.pow(self.alpha)
        p_j = p_j / p_j.sum()
        weights = (len(self.memory) * p_j).pow(-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_inc)
        
        return (states, actions, rewards, next_states, dones, weights, entries)
        
    def update(self, entries, errors):
        errors = errors.cpu().data.numpy()
        for entry, error in zip(entries, errors):
            entry[0] = error[0]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
