[//]: # (Image References)

[image1]: https://drive.google.com/uc?id=1rR6gnP-6y0EnEykJfVZ8CIsKcitOclTx "Trained Agent"

# Report

## Learning Algorithm

### The training loop

`main.py` contains the traditional training loop:

* initialize the environment
* for each episode
    * get the initial state from the environment
    * for each step in the current episode
        * get the next action from the agent
        * get the next state and reward from the environment
        * the agent observes the new experience and improve its model
        * verifiy if the agent achieved the goal

### The DQN algorithm
    
`agent.py` contains the DQN/Double DQN algorithms. Each time the training loop observes a new experience (state, action, reward, next state, done) the method `step` is invoked. The experience is stored into the replay buffer, and if the buffer contains at least batch size entries, a new batch of experience is sampled. 

The `learn` method  trains both the local and target networks based on such experiences. For the original DQN algorithm, the next Q-values are obtained directly from the target network. For the Double DQN algorithm, the action selection and evaluation are performed in different steps. First the action is obtained from the local network, it is the index of the max Q-value. Then, the evaluation of such action is obtained from the  target network. It is the corresponding Q-value of the previously selected action index. For both cases, the reward and discount factor gamma are applied to the target Q-value.

Finally, the TD error is computed as the difference between the target and current Q-values, and the weighted loss is calculated according to the configured replay buffer strategy.

### The replay Buffer

`memory.py` holds the implementation for the memory buffers strategies. The original DQN algorithm proposes a uniform sampling buffer with the objective of training the model by first storing experiences in the buffer, and then replaying a batch of experiences from it in a subsequent step. The expectation is to reduce the correlation of such observations, which leads to a more stable training procedure. The implementation of this strategy is defined in the ReplayBuffer class.

An interesting strategy is to control which experiences to sample from the buffer in order to maximize the learning. A possible solution is to get  experiences with higher TD error. Considering that higher TD errors provide more aggressive gradients to the network, it would approximate the optimal Q -value function faster. This is called prioritized experience replay, and it is implemented by the PrioritizedReplayBuffer class. 

### The neural network

`model.py` implements the neural network architecture. It consists of 4 Multilayer perceptron (MLP) layers. Each layer uses the RELU action function, except the last one, which has dimension equivalent to the number of actions. Each output unit represents the Q-value for that particular action. The table below shows the complete network model configuration:

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 37    | 128    | RELU       |
| 2     | linear | 128   | 64     | RELU       |
| 3     | linear | 64    | 16     | RELU       |
| 4     | linear | 16    | 4      | -          |

## Results

The environment is solved in 478 episodes. The following graph shows the reward progression of the last 100 episodes where the agent achieved +13 points.

![Trained Agent][image1]

## Ideas for Future Work

The submission has concrete future ideas for improving the agent's performance.

## References

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
