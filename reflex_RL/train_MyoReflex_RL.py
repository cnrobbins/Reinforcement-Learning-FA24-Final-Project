import ReflexInterface_RL
import numpy as np
from typing import Callable

import gymnasium
# import skvideo.io
import argparse
from datetime import datetime

from gymnasium.envs.registration import register

from stable_baselines3 import PPO, A2C, DDPG, DQN, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import set_random_seed

import os
import skvideo.io

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Vec env notes: save_freq = max(save_freq // n_envs, 1)
def switch_case(algo, environ, params_0, control_mode):

    if input_args.net_arch_val is not None:
        network_arch = list(np.array(input_args.net_arch_val, dtype=np.int32))
    else:
        network_arch = [128,128]

    if algo == 'PPO':
        # Default params
        return PPO("MlpPolicy", environ, learning_rate=0.001, n_steps=2048, use_sde=input_args.use_sde, policy_kwargs = dict(net_arch=network_arch), verbose=1, tensorboard_log=os.path.join(input_args.save_path, 'logs', 'tb_PPO'))
        
        # GaitNet params
        #return PPO("MlpPolicy", environ, learning_rate=0.0001, n_steps=4096, batch_size=1024, n_epochs=4, gae_lambda=0.99, target_kl=0.01, use_sde=input_args.use_sde, policy_kwargs = dict(net_arch=network_arch), verbose=1, tensorboard_log=os.path.join(input_args.save_path, 'logs', 'tb_PPO'))
        
        # Wang 2019 params
        # return PPO("MlpPolicy", environ, learning_rate=0.001, n_steps=50, batch_size=32, n_epochs=4, target_kl=0.01, gamma=0.96, use_sde=input_args.use_sde, policy_kwargs = dict(net_arch=network_arch), verbose=1, tensorboard_log=os.path.join(input_args.save_path, 'logs', 'tb_PPO'))        
    elif algo == 'A2C':
        return A2C("MlpPolicy", environ, learning_rate=0.0003, verbose=1)
    
    elif algo == 'DQN':
        return DQN("MlpPolicy", environ, learning_rate=0.0001, verbose=1)
    elif algo == 'SAC':
    
        if input_args.multi:
            print(f"Switching to non-episodic training for vec_env")
            model = SAC("MlpPolicy", environ, learning_rate=0.0003, learning_starts=2000, verbose=1, use_sde=input_args.use_sde, policy_kwargs=dict(net_arch=network_arch), gradient_steps=1, tensorboard_log=os.path.join(input_args.save_path, 'logs', 'tb_PPO'))
        elif input_args.single:
            model = SAC("MlpPolicy", environ, learning_rate=0.0003, learning_starts=2000, verbose=1, use_sde=input_args.use_sde, policy_kwargs=dict(net_arch=network_arch), train_freq=(5, "episode"), gradient_steps=-1, tensorboard_log=os.path.join(input_args.save_path, 'logs', 'tb_PPO'))
        
        return model
    
    elif algo == 'TD3':
        if input_args.multi:
            print(f"Cannot use vec_env with TD3, switching to single env")
            environ = gymnasium.make('MyoReflex_RL-v0', reflex_params=params_0, init_pose='walk', dt=0.01, mode=control_mode, tgt_field_ver=0, episode_limit=input_args.ep_limit)
        
        n_actions = environ.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3("MlpPolicy", environ, learning_rate=0.001, action_noise=action_noise, verbose=1)
    
    elif algo == 'DDPG':
        n_actions = environ.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return DDPG("MlpPolicy", environ, learning_rate=0.001, action_noise=action_noise, verbose=1)

def load_past_model(algo, environ, zipped_file):
    if algo == 'PPO':
        return PPO.load(zipped_file, env=environ, tensorboard_log=os.path.join(input_args.save_path, 'logs'))
    else:
        print('Not implemented yet')
        

def make_env(env_id: str, rank: int, env_args: dict, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gymnasium.make(env_id, render_mode=None, reflex_params=env_args['reflex_params'], 
                             init_pose=env_args['init_pose'], dt=env_args['dt'], mode=env_args['mode'], 
                             tgt_field_ver=env_args['tgt_field_ver'], episode_limit=env_args['episode_limit'], 
                             reward_wt=env_args['reward_wt'], target_vel=env_args['target_vel'])
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# From SB3 documentation (https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule)
def lr_linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

#---------- Arguments ----------
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()

parser.add_argument("-clu", "--cluster", action="store_true", help="Flag for script on cluster or local machine")
parser.add_argument("--move_dim", type=int, help="(int) Whether the model operates in 2D or 3D")
parser.add_argument("--tgt_vel", type=float, nargs='+', help="(float) X and Y target velocities")
parser.add_argument('--obs_param', required=False, nargs='+', help="Pass in a list of additional obs fields to extract")
parser.add_argument("--rew_type", type=int, help="(int) 0 for old reward func, 1 for new reward func")
parser.add_argument("--stim_mode", required=False, help="(String) reflex: modulate reflex gains, muscle: modulate muscle stim")
parser.add_argument("--tgt_vel_mode", required=False, help="(String) constant: constant target vel for each ep, sine: sinusoidal velocity in each ep")
parser.add_argument('--sine_vel_args', required=False, type=float, nargs='+', help="Parameters for sine curve: min, max vel and period")
parser.add_argument("--delta_mode", required=False, help="(String) reflex: realtime mode at 0.01 sec/timestep, delayed: 30 hz mode at 0.03 sec/timestep")
parser.add_argument("--delta_control_mode", required=False, help="(String) sym or asym: Default is (asym) control, which controls the gain of each leg individually")


parser.add_argument("--rl_algo", help="(String) RL policy to use")
parser.add_argument("--timestep", type=int, help="(int) Number of training timesteps")
parser.add_argument("--ep_limit", type=int, help="(int) Episode limit in no. of timesteps")
# parser.add_argument("--tgt_ver", type=int, help="(int) Version of velocity tgt field")
parser.add_argument("--chk_freq", type=int, help="(int) Checkpoint Frequency (in timesteps)")
parser.add_argument("--eval_freq", type=int, help="(int) Evaluation Frequency (in timesteps)")
parser.add_argument("--seed", type=int, help="(int) Seed for the environment")

group.add_argument("--single", action="store_true", help="Single env training")
group.add_argument("--multi", action="store_true", help="Enable vec_env training. Only works for certain RL policies")

# Conditional arguments
parser.add_argument("--vec_num", required=False, type=int, help="(int) Number of parallel environments, only works when --multi flag is True")
parser.add_argument("--use_sde", required=False, action="store_true", help="Flag to use gSDE, only works for PPO and SAC")
parser.add_argument('--rew_wt_args', required=False, nargs='+', help="Pass in a list of reward items for calculation. Must also specify rew_wt_val")
parser.add_argument('--rew_wt_val', required=False, type=float, nargs='+', help="Pass in a list of weights for reward items")
parser.add_argument('--net_arch_val', required=False, type=float, nargs='+', help="Pass in a list of desired network architecture")

# File loading/saving arguments
parser.add_argument("--param_path", required=False, help="(String) Path of param file, takes the first file in the directory")
parser.add_argument("--save_path", required=False, help="(String) Path to save outputs")
parser.add_argument("--model_path", required=False, help="(String) Path to load zipped file")

input_args = parser.parse_args()
#---------- End Arguments ----------

warnings.filterwarnings("ignore", category=DeprecationWarning)

register(
    id="MyoReflex_RL-v0",
    entry_point="ReflexInterface_RL:ReflexEnv",
    max_episode_steps=input_args.ep_limit,
)

print('Registration successful')

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    start_time = datetime.now()
    print(f"Training Started: {start_time}")

    #---------- Environment initialiations ----------
    if input_args.move_dim == 2:
        flag_ctrl_mode = '2D' # use 2D
        param_num = 47 + 8 + 18
    elif input_args.move_dim == 3:
        flag_ctrl_mode = '3D'
        param_num = 59 + 12 + 22

    if input_args.param_path is not None:
        files = os.listdir(input_args.param_path)
        files_txt = [i for i in files if i.endswith('.txt')]
        params_0 = np.loadtxt( os.path.join(input_args.param_path, files_txt[0]) )
    else:
        params_0 = np.ones(param_num,)

    #env = ReflexWrapper.ReflexEnv(mode=flag_ctrl_mode, init_pose='walk', episode_limit=input_args.ep_limit, tgt_field_ver=input_args.tgt_ver)
    #env.set_control_params(params_0)

    # Process reward item and weights
    if input_args.rew_wt_args is not None and input_args.rew_wt_val is not None:
        rew_and_wts = dict(zip(input_args.rew_wt_args, input_args.rew_wt_val) )
    else:
        rew_and_wts = None

    if input_args.tgt_vel is not None:
        tgt_vel = np.round(list(np.array(input_args.tgt_vel, dtype=np.float32)), 2)
    else:
        tgt_vel = np.array([-1,-1])

# parser.add_argument("--tgt_vel_mode", required=False, help="(String) constant: constant target vel for each ep, sine: sinusoidal velocity in each ep")
# parser.add_argument('--sine_vel_args', required=False, type=float, nargs='+', help="Parameters for sine curve: min, max vel and period")

    if input_args.obs_param is None:
        obs_param = None
    else:
        obs_param = input_args.obs_param
        
    if input_args.stim_mode is None:
        stim_mode = 'reflex' #default to reflex

    if input_args.tgt_vel_mode is None:
        tgt_vel_mode = 'constant' #default to constant vel
    else:
        tgt_vel_mode = input_args.tgt_vel_mode

    if input_args.sine_vel_args is not None:
        # currently hard-coded sequence of values
        # 1 - min vel
        # 2 - max vel
        # 3 - period
        temp_in = np.round(list(np.array(input_args.sine_vel_args, dtype=np.float32)), 2)
        sine_vel_args = dict(sine_min=temp_in[0], sine_max=temp_in[1], sine_period=temp_in[2])
    else:
        sine_vel_args = None

    if input_args.delta_mode is None:
        delta_mode = 'delayed'
    else:
        delta_mode = input_args.delta_mode

    if input_args.delta_control_mode is None:
        delta_control_mode = 'asym'
    else:
        delta_control_mode = input_args.delta_control_mode

    if input_args.rew_type is None:
        rew_type = 0
    else:
        rew_type = input_args.rew_type

    if input_args.single:
        train_env = gymnasium.make('MyoReflex_RL-v0', render_mode=None, reflex_params=params_0, init_pose='walk', dt=0.01, 
                                   mode=flag_ctrl_mode, 
                                   episode_limit=input_args.ep_limit, 
                                   reward_wt=rew_and_wts, target_vel=tgt_vel, 
                                   obs_param=obs_param, rew_type=rew_type, stim_mode=stim_mode, 
                                   tgt_vel_mode=tgt_vel_mode, sine_vel_args=sine_vel_args, delta_mode=delta_mode, delta_control_mode=delta_control_mode)
    elif input_args.multi:
        env_id = "MyoReflex_RL-v0"
        num_cpu = input_args.vec_num  # Number of processes to use
        # Create the vectorized environment
        #train_env = SubprocVecEnv([make_env(env_id, i, dict(reflex_params=params_0, init_pose='walk', dt=0.01, mode=flag_ctrl_mode, tgt_field_ver=input_args.tgt_ver, episode_limit=input_args.ep_limit, reward_wt=rew_and_wts, target_vel=tgt_vel)) for i in range(num_cpu)])
        train_env = make_vec_env(env_id, n_envs=num_cpu, seed=input_args.seed, 
                            env_kwargs=dict(render_mode=None, reflex_params=params_0, init_pose='walk', dt=0.01, mode=flag_ctrl_mode, 
                                            episode_limit=input_args.ep_limit, 
                                            reward_wt=rew_and_wts, target_vel=tgt_vel, obs_param=obs_param, rew_type=rew_type, stim_mode=stim_mode, 
                                            tgt_vel_mode=tgt_vel_mode, sine_vel_args=sine_vel_args, delta_mode=delta_mode, delta_control_mode=delta_control_mode), 
                                            vec_env_cls=SubprocVecEnv)


    # ----- Call backs -----
    # Save a checkpoint every X steps
    checkpoint_callback = CheckpointCallback(
    save_freq=input_args.chk_freq,
    save_path=os.path.join(input_args.save_path, 'logs'),
    name_prefix=str(input_args.rl_algo),
    save_replay_buffer=False,
    save_vecnormalize=False,
    )

    eval_env = make_vec_env('MyoReflex_RL-v0', n_envs=10, seed=input_args.seed, 
                            env_kwargs=dict(render_mode=None, reflex_params=params_0, init_pose='walk', dt=0.01, mode=flag_ctrl_mode, 
                                            episode_limit=input_args.ep_limit, 
                                            reward_wt=rew_and_wts, target_vel=tgt_vel, obs_param=obs_param, rew_type=rew_type, stim_mode=stim_mode, 
                                            tgt_vel_mode=tgt_vel_mode, sine_vel_args=sine_vel_args, delta_mode=delta_mode, delta_control_mode=delta_control_mode) )

    #eval_env = gymnasium.make('MyoReflex_RL-v0', reflex_params=params_0, init_pose='walk', dt=0.01, mode=flag_ctrl_mode, tgt_field_ver=input_args.tgt_ver, episode_limit=input_args.ep_limit, reward_wt=rew_and_wts, target_vel=tgt_vel)
    #eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(input_args.save_path, 'logs', 'tb_PPO'),
                                log_path=os.path.join(input_args.save_path, 'logs', 'tb_PPO'), eval_freq=input_args.eval_freq, n_eval_episodes=50,
                                deterministic=True, render=False)

    list_callback = CallbackList([checkpoint_callback, eval_callback])

    #---------- Training loop ----------
    print("Begin Training")
    if input_args.model_path is not None:
        
        files = os.listdir(input_args.model_path)
        files_zip = [i for i in files if i.endswith('.zip')]
        
        print(f"Continue training with: {files_zip[0]}")
        model = load_past_model(input_args.rl_algo, train_env, os.path.join(input_args.model_path, files_zip[0]))
        model.learn(total_timesteps=input_args.timestep, callback=list_callback, tb_log_name="RL_train", reset_num_timesteps=False)
    else:
        print("New training")
        model = switch_case(input_args.rl_algo, train_env, params_0, flag_ctrl_mode)
        model.learn(total_timesteps=input_args.timestep, callback=list_callback, tb_log_name="RL_train")
    
    model.save(os.path.join(input_args.save_path, f"{input_args.rl_algo}_{datetime.now().strftime('%Y%b%d_%H%M')}_{flag_ctrl_mode}_TrainedModel"))

    print("Training Done")
    end_time = datetime.now()
    elasped = end_time - start_time
    print(f"Elasped time (minutes) - {elasped.total_seconds()/60}")
    print(f"Training complete - {datetime.now()}")
