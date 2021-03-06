# -*- coding: utf-8 -*-
'''
Created on Tue Feb 7 12:16:30 2017

@author: zmarinho, ahefny
'''
from __future__ import print_function
import argparse
import numpy as np
from distutils.dir_util import mkpath
import json
from rpsp.run.stats_test import run_Nmodel
from rpsp import globalconfig
from rpsp.rpspnets.psr_lite.utils.log import Logger

class LoadFromJSONFile (argparse.Action):
    """
    Load a parameter file from a json configuration
    """
    def __call__(self, parser, namespace, values, option_string = None):
        """
        Load arguments parser with namespace and corresponding values
        @param parser: Argument parser from argparse
        @param namespace: attribute namespace
        @param values: data with keys and values for each command line argument
        @param option_string:
        @return: None simply set the name space with elements in values
        """
        with values as f:
            data = json.loads(f.read())
            for (k,v) in data.items():                
                setattr(namespace, k, v)
            setattr(namespace, 'original_tfile', data['tfile'])
            setattr(namespace, 'tfile', '')


def add_boolean_option (parser, arg_name, default=False, false_name=None, help=None, help_false=None):
    """
    Create boolean option for argparse.  With default Name and default value and false name if value is False for disambiguation.
    @param parser: Argparser
    @param arg_name:  default argument name
    @param default: default value
    @param false_name:  Name for False boolean value
    @param help:
    @param help_false:
    @return:
    """
    if false_name is None:
        false_name = 'no_' + arg_name
    parser.add_argument('--' + arg_name, dest=arg_name, action='store_true', help=help)
    parser.add_argument('--' + false_name, dest=arg_name, action='store_false', help=help_false)
    parser.set_defaults(**{arg_name : default})
    
def get_parser():
    """
    Command line options parser.
    If run pre-specified model run config file of model in tests/
    @return: arguments
    """
    parser = argparse.ArgumentParser(description='RPSP network')

    parser.add_argument('--config', type=open, action=LoadFromJSONFile, \
                        help='A JSON file storing experiment configuration. This can be found in the tests directory')
    parser.add_argument('--datapath', type=str, \
                        help='Directory containing trained_models. Each pkl file contains matrices W2ext W2oo W2fut.')
    
    # Environment Options
    parser.add_argument('--env', type=str, required=False,  default='CartPole-v0', help='Gym environment. Check available environments in envs.load_environments.env_dict [CartPole-v0].')
    add_boolean_option(parser, 'addrwd', default=False, help='Add rewards to observation model [False].')
    add_boolean_option(parser, 'addobs', default=False, help='Add observations to predictive states, this are the augmented states in the paper [False].')
    parser.add_argument('--p_obs_fail', type=float, help='sensor failure probability for observation.')
    parser.add_argument('--T_obs_fail', type=int, help='max sensor failure time window for observation.')
    parser.add_argument('--obsnoise', type=float, default=0.0, help='standard deviation for noise in observation space [0].')
    parser.add_argument('--obs_latency', type=int, default=0, help='observation latency in steps [0].')
    parser.add_argument('--act_latency', type=int, default=0, help='action latency in steps [0].')
    add_boolean_option(parser, 'fullobs', default=False, help='Use fully observable environment state [False].')
    add_boolean_option(parser, 'normalize_act', default=True, help='Scale actions within bounds [True].')
    add_boolean_option(parser, 'normalize_obs', default=False, help='Scale obs mean avg [False].')
    add_boolean_option(parser, 'normalize_rwd', default=False, help='Scale rwds mean avg [False].')

    # Model Options
    parser.add_argument('--method', type=str, required=False, help='function to call.')
    parser.add_argument('--nh', nargs='+', type=int, default=[16],  help='number of hidden units. --nh L1 L2 ... number of hidden units for each layer [16].')
    parser.add_argument('--nL', type=int, default=1, help='number of layers. 0- for linear [1]')
    parser.add_argument('--nn_act', type=str, default='relu', help='Activation function for feed-forward netwroks (relu/tanh) [relu]')
    parser.add_argument('--fut', type=int, help='future window size')
    parser.add_argument('--past', type=int, help='past window size')
    parser.add_argument('--reg', type=float, default=0.01, help='uniform regularization constant [0.01]')
    parser.add_argument('--reg_s1a', type=float,   help='s1a regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_s1b', type=float,   help='s1b regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_s1c', type=float,   help='s1c regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_s1div', type=float,   help='s1_divide regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_ex', type=float,   help='Wext regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_oo', type=float,   help='Woo regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_pred', type=float,   help='pred regularization constant if not specified in reg for all.')
    parser.add_argument('--reg_filter', type=float,   help='filter regularization constant if not specified in reg for all.')
    parser.add_argument('--dim', type=int,  help='PSR dimension (latent space dim).')
    parser.add_argument('--Hdim', type=int, default=1000, help='high features dimension.')
    parser.add_argument('--fext', type=str, help='feature extractor to call.')
    parser.add_argument('--kw', type=int, default=50, help='kernel width percentile [default:50]')
    parser.add_argument('--min_std', type=float, default=0.0, help='minimum policy std. [0.0]')
    parser.add_argument('--gclip', type=float, default=10.0, help='gradient clipping [10.0]')
    parser.add_argument('--r_max', type=float, default=10000.0, help='probability ratio limit max for TRPO and AltOp [10000]')
    
    add_boolean_option(parser, 'random_start', default=False, help='Start the psr with Random parameters [False]')
    parser.add_argument('--psr_iter', type=int, default=5, help='Number of rffpsr/rffpsr_rnn conjugate grad iterations [5]')
    parser.add_argument('--psr_state_norm', type=str, default='I', help="'I' identity, 'l2' l2 normalization, 'of' simple feature] [I]")
    parser.add_argument('--psr_smooth', type=str, default='I', help="'I' no op, 'interp_0.9' convex interpolation, 'search' do a search direction] [I]")
    parser.add_argument('--psr_cond', type=str, default='kbr', help="rff psr state update ['kbr':kernel bayes rule, 'kbrcg' KBR with conjugate gradient, 'kbrMIA' matrix inverse approximation vi Neumann Series, 'I' ignore Coo v= t_obs_feat] [default:kbr]")
  
    # PSR Refinement Options
    parser.add_argument('--refine', type=int, default=0, help='number of model refinment iterations at initialization [0]')
    parser.add_argument('--valratio', type=float, help='training ratio. rest is for validation in refinment at initialization')
    parser.add_argument('--roptimizer', type=str, default='adam', help='Optmizer for PSR refinement at initialization (adam, sgd, rmsprop, adadelta, adagrad).[adam]')
    parser.add_argument('--rstep', type=float, default=0.1, help='gradient descent step size [0.1]')
    parser.add_argument('--valbatch', type=int, default=0, help='Refinement batch size at initialization')
    parser.add_argument('--minrstep', type=float,default=1e-5, help='minimum refinement step [1e-5]')
    
    # Data Collection Options
    parser.add_argument('--numtrajs', default=0, type=int, required=False,  help='number of trajectories per iteration [0]')
    parser.add_argument('--numsamples', default=0, type=int, required=False,  help='number of samples per iteration (either select numtrajs or numsamples)')
    parser.add_argument('--mintrajlen', default=0, type=int, required=False,  help='minimum number of samples per iteration for PSR filter (biggert than past+future+2 windows)')
    #parser.add_argument('--rtmax', type=int, help='max number of retraining trajectories')
    parser.add_argument('--len', type=int, required=False, help='Maximum of nominal trajectory length')    
    parser.add_argument('--leni', type=int, help='Maximum of initial trajectory length. Sample until this value is reached')
    parser.add_argument('--initN', default=0, type=int, help='number of initial trajectories used to initialize model/policy')    
    parser.add_argument('--initS', default=0, type=int, help='number of initial samples used to initialize model/policy')    
    
    # Training Options
    parser.add_argument('--iter', type=int, required=False, default=500, help='number of training iterations')    
    parser.add_argument('--lr', type=float, required=False, default=1e-2, help='Learning rate for VR Reinforce')        
    parser.add_argument('--grad_step', type=float, default=1e-2, help='Learning rate for VR Reinforce in AltOp')
    parser.add_argument('--trpo_step', type=float, help='TRPO Step size in ALtOp')
    parser.add_argument('--cg_opt', type=str, default='adam', help='gradient optimizer. Options: adam, adadelta,adagrad,RMSProp,sgd (default:adam)')
    parser.add_argument('--wpred', type=float, default=1.0, help='weight on predictive error policy')
    parser.add_argument('--wdecay', type=float, default=1.0, help='decay weight if 1.0 does not decay over time [1.0]')
    parser.add_argument('--wrwd', type=float, default=1.0, help='weight on rewards for joint objective')
    parser.add_argument('--wkl', type=float, default = 1.0, help='weight on KL psr 1 step difference ')
    parser.add_argument('--wrwd_only', type=float, default=1.0, help='weight on rewards for individual objective')
    parser.add_argument('--repeat', type=int, default=1, help='number of times run each model to get stats results')
    add_boolean_option(parser, 'h0', default=True, help=' Optimize intial predictive state [True]')
    parser.add_argument('--pi_exp', type=str, default='None', help='exploration strategy [ gauss, None]')
    parser.add_argument('--seed', type=int, default=0, help='experiment seed [0: default]')    
    parser.add_argument('--b', type=str, default='psr', help='type of baseline for PSR network [psr:default, obs,AR, none]')
    parser.add_argument('--vr', type=str, default='VRpg', help='Reinforce method  [default:VRpg], VRpg, TRPO')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for computing value function (cost-to-go)')
    parser.add_argument('--discount', type=float, default=1.0, help='discount factor for adam over iterations')
    add_boolean_option(parser, 'norm_g', default=True, help='Normalize policy gradients. [TRUE]')    
    parser.add_argument('--filter_w', type=int, default=1, help='Filtering model to run environment FMi[default:1]')
    parser.add_argument('--beta_lr', type=float, default=0.0, help='Update learning rate via variance normalization. exp.averaging ratio [0.0]')    
    parser.add_argument('--beta', type=float, default=0.1, help='Exp.averaging ratio for variance relative weights [0.1]')    
    parser.add_argument('--load_reactive', type=str, default='', help='Path to pickle file load reactive policy parameters')
    parser.add_argument('--decay1', type=float, default=0.0, help='decrease rate for loss1 importance')
    parser.add_argument('--decay2', type=float, default=0.0, help='decrease rate for loss2 importance')
    parser.add_argument('--threshold_var', type=float, default=0.0, help='threshold percentile for variance normalization')
    parser.add_argument('--var_clip', type=float, default=0.0, help='clip variance normalization [1/val, val]')
    add_boolean_option(parser, 'fix_psr', default=False, help='Fix PSR parameters. [FALSE]')    
    parser.add_argument('--hvec', type=str, default='exact', help='Hessian-vector multiplication method. (exact,fd)')
    
    # Output Options
    parser.add_argument('--tfile', type=str, default='results/', help='Directory containing test data.')
    parser.add_argument('--monitor', type=str, help='monitor file.')
    parser.add_argument('--vrate', type=int, default=1, help='Number of iterations after which a video is saved when monitoring is enabled')
    parser.add_argument('--irate', type=int, default=100000, help='Number of iterations after which a trajectory image is saved')
    parser.add_argument('--prate', type=int, default=50, help='Number of iterations after which results are pickled')
    add_boolean_option(parser, 'log', default=False, help='Report mse after updates and log UTU print slower')
    parser.add_argument('--logfile', type=str, help='Path to pickle file to save the output of the Logger')
    parser.add_argument('--loadfile', type=str,default='', help='Path to pickle file to save the output of the Logger')
    add_boolean_option(parser, 'render', default=False, help='Report mse after updates and log UTU print slower')
    
    # Debug Options
    parser.add_argument('--dbg_len', nargs=2, type=int, help='Specifies a range of traj lengths to be chosen at random')
    add_boolean_option(parser, 'dbg_nobatchpsr', false_name='dbg_batchpsr', default=True, help='Do not use batched PSR updates')
    add_boolean_option(parser, 'mask_state', default=False, help='set predictive states to 0.[Default:False]')    
    parser.add_argument('--dbg_reward', default=0.0, type=float, help='Control cost for Envwrapper for reward shapping. [Default:0.0 same as default openAI] ')
    parser.add_argument('--powerg', default=2., type=float, help='Control cost for Envwrsquash gradietn to -1,1.[Default:False]')
    add_boolean_option(parser, 'verbose', default=False, help='Print additional logging info.[Default:False]')

    # Start with exploration trajectories
    parser.add_argument('--exp_trajs', type=str, help='pickle file with exploration trajectories')
    return parser
       
if __name__ == '__main__': 
    parser = get_parser()       
    args = parser.parse_args()
    test_file = args.tfile + '/'
    mkpath(test_file)    
    setattr(args, 'file', test_file)
    print(test_file)
    #setattr(args, 'logfile', test_file+'log.log')
    json.dump(args.__dict__, open(test_file+'params','w'))
    
    args.rng = np.random.RandomState(args.seed)
    globalconfig.vars.args = args
   

    if args.logfile is not None:
        Logger.instance().set_file(args.logfile)
    run_Nmodel(args, test_file, N=args.repeat, loadfile=args.loadfile)
    Logger.instance().stop()
    
    
