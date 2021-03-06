#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny, zmarinho
"""
from rpsp.envs.environments import GymEnvironment, PartiallyObservableEnvironment, ContinuousEnvironment, \
    NoisyEnvironment, LatencyEnvironment, NormalizingEnvironment, ExtendedEnvironment, RewardShapingEnv, Renderer, \
    SensorFailureEnvironment
from rpsp.envs.simulators import CartpoleContinuousSimulator, PendulumContinuousSimulator, AcrobotContinuousSimulator
import gym
import numpy as np


class structtype():
    pass

'''
Each create_env function takes command line arguments and creates 
environment, exploration model and exploration policy. The latter two are used
to generate initial trajectories.

A function can return None for model and policy. In this case an observable model
and a random Gaussian policy are used.
'''


def monitor_env(args, env, env_name, entry_point):
    from gym.envs.registration import register
    from gym.wrappers import Monitor
    mon_name = (env_name + '-trial%d-v1' % args.trial, env_name)[args.monitor is None]
    register(
        id=mon_name,
        entry_point=entry_point,
        tags={'wrapper_config.TimeLimit.max_episode_steps': args.len},
        reward_threshold=1e5,
    )
    print('monitor wrapper')
    monitor = Monitor(env.env,
                      directory=args.flname + args.monitor,
                      force=True, video_callable=lambda it: it % int(args.vrate) == 0,
                      write_upon_reset=True)

    env = GymEnvironment(monitor, discrete=False, rng=args.rng)
    return env


def create_gym_env(args, env_name, entry_point):
    env = gym.make(env_name)
    env.seed(int(args.seed) + args.trial)
    if args.monitor is not None:
        print('using monitor env')
        env = monitor_env(args, env, env_name, entry_point)
    else:
        env = GymEnvironment(env, discrete=False, rng=args.rng)
    return env


def create_partial_environment(args, env, part_obs, full_obs=None):
    if not args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array(part_obs))
    elif full_obs is not None:
        env = PartiallyObservableEnvironment(env, np.array(full_obs))
    return env


def create_gym_mujoco_env(args, env_name, entry_point, part_obs, full_obs=None):
    env = create_gym_env(args, env_name, entry_point)
    env = create_partial_environment(args, env, part_obs, full_obs=full_obs)
    return env


def create_continuous_classic_env(args, env_name, entry_point, part_obs, full_obs=None, sim=None):
    env = create_gym_env(args, env_name, entry_point)
    if sim is not None:
        env = ContinuousEnvironment(env, sim, qpos_dim=part_obs, qvel_dim=list(set(full_obs) - set(part_obs)))
    env = create_partial_environment(args, env, part_obs, full_obs=full_obs)
    return env



'''
MUJOCO ENVIRONMENTS
'''
def create_env_swimmer(args):
    env = create_gym_mujoco_env(args, args.env, 'gym.envs.mujoco:SwimmerEnv', [0, 1, 2])
    return env


def create_env_hopper(args):
    env = create_gym_mujoco_env(args, 'Hopper-v1', 'gym.envs.mujoco:HopperEnv', [0, 1, 2, 3, 4])
    return env


def create_env_walker(args):
    env = create_gym_mujoco_env(args, 'Walker2d-v1', 'gym.envs.mujoco:Walker2d', [0, 1, 2, 3, 4, 5, 6, 7])
    return env


def create_env_ant(args):
    env = create_gym_mujoco_env(args, 'Ant-v1', 'gym.envs.mujoco:AntEnv', np.arange(8, 16), np.arange(16))
    return env


def create_env_Mcartpole(args):
    env = create_gym_mujoco_env(args, 'InvertedPendulum-v1', 'gym.envs.mujoco:InvertedPendulum', [0, 1])
    return env


def create_env_reacher(args):
    env = create_gym_mujoco_env(args, 'Reacher-v1', 'gym.envs.mujoco:Reacher', [2, 3, 4, 5])
    return env


def create_env_doublependulum(args):
    env = create_gym_mujoco_env(args, 'InvertedDoublePendulum-v1', 'gym.envs.mujoco:InvertedDoublePendulum', [0, 1, 2],
                                [0, 1, 2, 5, 6, 7])
    return env

'''
CLASSIC CONTROL ENVIRONMENTS
'''

def create_classic_env_cartpole(args):
    env = create_continuous_classic_env(args, 'CartPole-v0', 'gym.envs.classic_control.cartpole:CartPoleEnv',
                                      [0, 2], [0, 2, 1, 3], sim=CartpoleContinuousSimulator())
    return env



env_dict = {'Swimmer-v1': create_env_swimmer,
            'Walker2d-v1': create_env_walker,
            'Hopper-v1': create_env_hopper,
            'CartPole-v1': create_env_Mcartpole,
            'CartPole-v0': create_classic_env_cartpole,
            'Reacher-v1':create_env_reacher,
            }


def load_environment(args):
    env = env_dict[args.env](args)
    return env
