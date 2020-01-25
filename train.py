from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np
from chainer import cuda

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers
import json
from q_functions import QFunction, DuelingQFunction
from q_functions import NoisyQFunction, NoisyDuelingQFunction
from q_functions import DistributionalQFunction, DistributonalDuelingQFunction
from q_functions import NoisyDistributionalQFunction, NoisyDistributonalDuelingQFunction
from value_buffer import ValueBuffer
from eva_replay_buffer import EVAReplayBuffer, EVAPrioritizedReplayBuffer
from agents import EVA, EVADoubleDQN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--steps', type=int, default=10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--replay-start-size', type=int, default=4 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=30)
    parser.add_argument('--update_interval', type=int, default=4)
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--periodic_steps', type=int, default=20,
                        help='backup insert period')
    parser.add_argument('--value_buffer_neighbors', type=int, default=5,
                        help='Number of k')
    parser.add_argument('--lambdas', type=float, default=0.4,
                        help='Number of λ')
    parser.add_argument('--replay_buffer_neighbors', type=int, default=10,
                        help='Number of M')
    parser.add_argument('--len_trajectory', type=int, default=50,
                        help='max length of trajectory(T)')
    parser.add_argument('--replay_buffer_capacity', type=int, default=500000,
                        help='Replay Buffer Capacity')
    parser.add_argument('--value_buffer_capacity', type=int, default=2000,
                        help='Value Buffer Capacity')
    parser.add_argument('--minibatch_size', type=int, default=48,
                        help='Training batch size')
    parser.add_argument('--target_update_interval', type=int, default=2000,
                        help='Target network period')
    parser.add_argument('--LRU', action='store_true', default=False,
                        help='Use LRU to store in value buffer')
    parser.add_argument('--prioritized_replay', action='store_true', default=False)
    parser.add_argument('--dueling', action='store_true', default=False,
                        help='use dueling dqn')
    parser.add_argument('--noisy_net_sigma', type=float, default=None,
                        help='NoisyNet explorer switch. This disables following options: '
                        '--final-exploration-frames, --final-epsilon, --eval-epsilon')
    parser.add_argument('--num_step_return', type=int, default=1)
    parser.add_argument('--agent', type=str, default='EVA', 
                        choices=['EVA', 'DoubleEVA', 'CategoricalEVA', 'CategoricalDoubleEVA'])

    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed
    if args.lambdas == 0 or args.lambdas == 1:
        if args.agent == 'EVA':
            agent_name = 'DQN'
        elif args.agent == 'DoubleEVA':
            agent_name = 'DoubleDQN'
        elif args.agent == 'CategoricalEVA':
            agent_name = 'CategoricalDQN'
        elif args.agent == 'CategoricalDoubleEVA':
            agent_name = 'CategoricalDoubleDQN'
    else:
        agent_name = args.agent

    if args.dueling == True:
        q = 'Dueling'
    else:
        q = 'DQN'

    if args.prioritized_replay == True:
        p = 'PER'
    else:
        p = 'ER'

    if args.noisy_net_sigma is not None:
        n = 'Noisy'
    else:
        n = 'Egreedy'

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir,
        time_format='{}/{}/{}/{}step/{}/{}/seed{}/%Y%m%dT%H%M%S.%f'.format(agent_name, q, p, args.num_step_return,
                                                                           n, args.env, args.seed))
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=None),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, 0.001)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    if args.gpu >= 0:
        xp = cuda.cupy
    else:
        xp = np

    n_actions = env.action_space.n

    n_history = 4

    if args.agent == 'CategoricalEVA' or 'CategoricalDoubleEVA':
        n_atoms = 51
        v_max = 10
        v_min = -10
        if args.noisy_net_sigma is not None:
            if args.dueling:
                q_func = NoisyDistributonalDuelingQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU,
                                                            n_atoms=n_atoms, v_min=v_min, v_max=v_max, n_hidden=256,
                                               lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                               num_neighbors=args.value_buffer_neighbors, sigma=args.noisy_net_sigma)
            else:
                q_func = NoisyDistributionalQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU,
                                                      n_atoms=n_atoms, v_min=v_min, v_max=v_max, n_hidden=256,
                                        lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                        num_neighbors=args.value_buffer_neighbors, sigma=args.noisy_net_sigma)
            explorer = explorers.Greedy()
        else:
            if args.dueling:
                q_func = DistributonalDuelingQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU,
                                                       n_atoms=n_atoms, v_min=v_min, v_max=v_max, n_hidden=256,
                                          lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                          num_neighbors=args.value_buffer_neighbors)

            else:
                q_func = DistributionalQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU,
                                   n_atoms=n_atoms, v_min=v_min, v_max=v_max, n_hidden=256,
                                   lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                   num_neighbors=args.value_buffer_neighbors)

            explorer = explorers.LinearDecayEpsilonGreedy(
                start_epsilon=1.0, end_epsilon=0.01,
                decay_steps=10 ** 6,
                random_action_func=lambda: np.random.randint(n_actions))
    else:
        if args.noisy_net_sigma is not None:
            if args.dueling:
                q_func = NoisyDuelingQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU, n_hidden=256,
                                               lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                               num_neighbors=args.value_buffer_neighbors, sigma=args.noisy_net_sigma)
            else:
                q_func = NoisyQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU, n_hidden=256,
                                        lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                        num_neighbors=args.value_buffer_neighbors, sigma=args.noisy_net_sigma)
            explorer = explorers.Greedy()
        else:
            if args.dueling:
                q_func = DuelingQFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU, n_hidden=256,
                                          lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                          num_neighbors=args.value_buffer_neighbors)

            else:
                q_func = QFunction(n_history, num_actions=n_actions, xp=xp, LRU=args.LRU, n_hidden=256,
                                   lambdas=args.lambdas, capacity=args.value_buffer_capacity,
                                   num_neighbors=args.value_buffer_neighbors)

            explorer = explorers.LinearDecayEpsilonGreedy(
                start_epsilon=1.0, end_epsilon=0.01,
                decay_steps=10 ** 6,
                random_action_func=lambda: np.random.randint(n_actions))


    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyperparameters as the Nature paper
    opt = optimizers.Adam(0.0001)
    opt.setup(q_func)

    if args.prioritized_replay:
        betasteps = (args.steps - args.replay_start_size) \
                    // args.update_interval
        rbuf = EVAPrioritizedReplayBuffer(
            args.replay_buffer_capacity, num_steps=args.num_step_return, key_width=256, xp=xp, M=args.replay_buffer_neighbors,
            T=args.len_trajectory,
            betasteps=betasteps)
    else:
        rbuf = EVAReplayBuffer(args.replay_buffer_capacity, num_steps=args.num_step_return, key_width=256, xp=xp,
                               M=args.replay_buffer_neighbors,
                               T=args.len_trajectory)


    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    def get_agent(agent):
        return {'EVA': EVA,
                'DoubleEVA': EVADoubleDQN,
                'CategoricalEVA': CategoricalEVA,
                'CategoricalDoubleEVA': CategoricalDoubleEVA}[agent]

    Agent = get_agent(args.agent)   

    agent = Agent(q_func, opt, rbuf, gamma=args.gamma,
                  explorer=explorer, gpu=args.gpu, replay_start_size=args.replay_start_size,
                  minibatch_size=args.minibatch_size, update_interval=args.update_interval,
                  target_update_interval=args.target_update_interval, clip_delta=True,
                  phi=phi,
                  target_update_method='hard',
                  soft_update_tau=args.soft_update_tau,
                  n_times_update=1, average_q_decay=0.999,
                  average_loss_decay=0.99,
                  batch_accumulator='mean', episodic_update=False,
                  episodic_update_len=16,
                  len_trajectory=args.len_trajectory,
                  periodic_steps=args.periodic_steps
                  )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None)
        print('n_episodes: {} mean: {} median: {} stdev {}'.format(
            eval_stats['episodes'],
            eval_stats['mean'],
            eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 30 evaluation episodes, each capped at 5 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=4500,
            logger=None)
        with open(os.path.join(args.outdir, 'bestscores.json'), 'w') as f:
            # temporary hack to handle python 2/3 support issues.
            # json dumps does not support non-string literal dict keys
            json_stats = json.dumps(stats)
            print(str(json_stats), file=f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))

if __name__ == '__main__':
    main()