# REVA
 
 EVAとRainbowを統合させる

 ## Ephemeral Value Adjustment
[Fast deep reinforcement learning using online adjustments from the past](https://arxiv.org/pdf/1810.08163.pdf)

 ## Rainbow
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf)

## TODO
- C51の実装
- Rainbowの完成版
- REVAの完成版

## Usage
- --doubledqn Double DQN使用
- --prioritized_replay PER使用
- --dueling dueling network用いる
- --noisy_net_sigma 探索にNoisy Nets(数値設定)
- --n_step_return 
- --lambdas 0か1でノンパラメトリックは全く使用しない
- --LRU value bufferのストア時にLRUを用いる

## Requirement
- chainerrl==0.6.0

## Note
- Resultsの構造
    - 手法(DDDQNなど)/Nstep/探索法/game名/seed値/...