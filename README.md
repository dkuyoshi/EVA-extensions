# REVA
 

Integrate Rainbow with Ephemeral Value Adjustment(EVA)

 ## Ephemeral Value Adjustment
[Fast deep reinforcement learning using online adjustments from the past](https://arxiv.org/pdf/1810.08163.pdf)

 ## Rainbow
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)


## Usage
- train.py

    Adapt to any combination of algorithms
    - --agent choose 'EVA', 'DoubleEVA', 'CategoricalEVA', 'CategoricalDoubleEVA'
    - --prioritized_replay Using PER
    - --dueling Using dueling network
    - --noisy_net_sigma Using Noisy Nets(Numeric settings)
    - --n_step_return Set to numeric
    - --lambdas Set to 0 or 1 to use non parametric at all(DQN, DoubleDQN... .etc)
    - --LRU Use LRU when storing at value buffer
    

- train_reva.py

    Train with Rainbow
    
    - --lambdas Set to 0 or 1 for using simple Rainbow

## Requirement
- chainerrl==0.6.0

## Note
- train.py
    - Constraction of results
        - Algorithms(DDDQN .etc)/Nstep/exploration/game/seed/...
- train_reva.py
    - Constraction of results
        - RAINBOW or REVA/game/seed...
