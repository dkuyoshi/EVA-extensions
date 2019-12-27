# REVA
 

Integrate Rainbow with EVA

 ## Ephemeral Value Adjustment
[Fast deep reinforcement learning using online adjustments from the past](https://arxiv.org/pdf/1810.08163.pdf)

 ## Rainbow
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)

## TODO
- Implementation of C51
- Complete version of Rainbow
- Complete version of REVA

## Usage
- --doubledqn Using Double DQN
- --prioritized_replay Using PER
- --dueling Using dueling network
- --noisy_net_sigma Using Noisy Nets(Numeric settings)
- --n_step_return 
- --lambdas Set to 0 or 1 to use non parametric at all
- --LRU Use LRU when storing at value buffer

## Requirement
- chainerrl==0.6.0

## Note
- Constraction of results
    - Algorithms(DDDQN .etc)/Nstep/exploration/game/seed/...