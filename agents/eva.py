from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()  # NOQA

import os
from logging import getLogger
import numpy as np

import chainer
from chainer import cuda
from chainerrl.agents import dqn
from chainerrl.misc.batch_states import batch_states
from chainerrl.recurrent import Recurrent
from chainerrl.replay_buffer import batch_experiences

import copy
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import state_kept
import time

class EVA(dqn.DQN):
    """EVA algorithm(Apply to DQN)"""
    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=40000,
                 minibatch_size=48, update_interval=1,
                 target_update_interval=100, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=100,
                 logger=getLogger(__name__),
                 batch_states=batch_states,
                 len_trajectory=50,
                 periodic_steps=20
                 ):
        super().__init__(q_function, optimizer, replay_buffer, gamma,
                         explorer, gpu, replay_start_size=replay_start_size,
                         minibatch_size=minibatch_size, update_interval=update_interval,
                         target_update_interval=target_update_interval, clip_delta=clip_delta,
                         phi=phi,
                         target_update_method=target_update_method,
                         soft_update_tau=soft_update_tau,
                         n_times_update=n_times_update, average_q_decay=average_q_decay,
                         average_loss_decay=average_loss_decay,
                         batch_accumulator=batch_accumulator, episodic_update=episodic_update,
                         episodic_update_len=episodic_update_len,
                         logger=logger,
                         batch_states=batch_states)

        self.last_embed = None
        self.len_trajectory = len_trajectory
        self.num_actions = self.model.num_actions
        self.periodic_steps = periodic_steps
        self.value_buffer = self.model.non_q
        self.current_t = 0

    def act_and_train(self, obs, reward):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi), eva=(len(self.value_buffer.embeddings) == self.value_buffer.capacity))
            q = float(action_value.max.array)
            greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]
            embed = cuda.to_cpu(self.model.get_embedding().array)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer

            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                embedding=self.last_embed[0],
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action
        self.last_embed = embed

        self.replay_updater.update_if_necessary(self.t)
        self.backup_store_if_necessary(embed, self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def backup_store_if_necessary(self, embedding, t):
        if self.model.lambdas == 0 or self.model.lambdas==1:
            return
        if (t % self.periodic_steps == 0) and (self.t >= self.replay_buffer.capacity):
            self.replay_buffer.update_embedding()
            trajectories = self.replay_buffer.lookup(embedding)
            batch_trajectory = [{'state': batch_states([elem[0]['state'] for elem in traject], self.xp, self.phi),
                                  'action': [elem[0]['action'] for elem in traject],
                                  'reward': [elem[0]['reward'] for elem in traject],
                                  'embedding': [elem[0]['embedding'] for elem in traject]
                                  } for traject in trajectories]

            qnp, embeddings = self._trajectory_centric_planning(batch_trajectory)
            self.value_buffer.store(embeddings, qnp)

    def _trajectory_centric_planning(self, trajectories):
        #atari
        embeddings = []
        batch_state = []
        for trajectory in trajectories:
            embeddings += trajectory['embedding']
            batch = self.xp.empty((self.len_trajectory, 4, 84, 84), dtype=self.xp.float32)
            batch[:len(trajectory['state'])] = trajectory['state']
            batch_state.append(batch)

        batch_state = self.xp.concatenate(batch_state, axis=0).astype(self.xp.float32)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            parametric_q = self.model(batch_state, eva=False)
            parametric_q = cuda.to_cpu(parametric_q.q_values.array).reshape((len(trajectories), self.len_trajectory , self.num_actions))

        q_value = []
        for qnp, trajectory in zip(parametric_q, trajectories):
            action = trajectory['action']
            reward = trajectory['reward']
            T = len(action)
            qnp = qnp[:T]
            Vnp = np.max(qnp[T - 1])
            for t in range(T - 2, -1, -1):
                qnp[t][action[t]] = reward[t] + self.gamma * Vnp
                Vnp = np.max(qnp[t])
            q_value.append(qnp)
            
        return self.xp.asarray(np.concatenate(q_value, axis=0).astype(np.float32)), self.xp.asarray(embeddings)
        
    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi), eva=(len(self.value_buffer.embeddings) == self.value_buffer.capacity))
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]
            embed = cuda.to_cpu(self.model.get_embedding().array)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q
        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        self.backup_store_if_necessary(embed, self.current_t)
        self.current_t += 1
        return action

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model.q_func)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model.q_func,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

    def stop_episode_and_train(self, state, reward, done=False):
        assert self.last_state is not None
        assert self.last_action is not None
        assert self.last_embed is not None

        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            embedding=self.last_embed[0],
            next_state=state,
            is_state_terminal=done,
        )

        self.replay_updater.update_if_necessary(self.t)
        self.backup_store_if_necessary(self.last_embed, self.t)
        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None
        self.last_embed = None
        self.current_t = 0

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

        self.replay_buffer.stop_current_episode()
