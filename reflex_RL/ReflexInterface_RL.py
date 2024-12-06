"""
implemented from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

from __future__ import division # '/' always means non-truncating division

import copy
import myosuite
import numpy as np
# import gym
import gymnasium
from gymnasium import spaces

# Note: Both gym and gymnasium packages are installed, be careful to differentiate them
import numpy as np
import skvideo.io
import os

# from v_tgt_field import VTgtField

from ReflexCtr_RL_11mus import MyoLocoCtrl_RL


class ReflexEnv(gymnasium.Env):

    ACT_RANGE = np.array([-1, 1])
    REF_GAIN_RANGE = np.array([-1, 1])
    reflexDataList = [
        'theta', 'dtheta', 'theta_f', 'dtheta_f',
        'pelvis_pos', 'pelvis_vel',
    ]

    legDatalist = [
        'load_ipsi',
        'talus_contra_pos',
        'talus_contra_vel',
        'phi_hip','phi_knee','phi_ankle',
        'dphi_hip','dphi_knee','alpha',
        'dalpha','alpha_f',
        'F_GLU','F_VAS','F_SOL','F_GAS','F_HAM','F_HAB',
    ]

    reflexOutputList = [
        'spinal_control_phase',
        'supraspinal_command',
        'moduleOutputs',
    ]

    # if 3D, also include hip adduction
    pose_key = ['pelvis_ty', 'pelvis_tilt', 
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l', 
                'vel_pelvis_tx', 
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l']
    pose_map = dict(zip(pose_key, range(len(pose_key))))

    init_act_key = ['GLU_r', 'HFL_r', 'HAM_r', 'RF_r', 'BFSH_r', 'GAS_r', 'SOL_r', 'VAS_r', 'TA_r', 
                    'GLU_l', 'HFL_l', 'HAM_l', 'RF_l', 'BFSH_l', 'GAS_l', 'SOL_l', 'VAS_l', 'TA_l', 
                    'HAB_r', 'HAD_r',
                    'HAB_l', 'HAD_l',
                    ]
    init_act_map = dict(zip(init_act_key, range(len(init_act_key))))

    mus_len_key = []
    mus_len_map = dict(zip(mus_len_key, range(len(mus_len_key))))

    CONTROL_PARAM = []
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        'alive_rew': 0,
        # 'action_penalty_zero': 1.0,
        #'action_penalty': 1.0,
        'truncated': -10,
        # 'footstep': 1,
        'effort': 0,
        'v_tgt': 1.0,
    }

    JNT_OPTIM = {}
    DEFAULT_INIT_MUSC = {}
    height_offset = 0
    SENSOR_DATA = {'body':{}, 'r_leg':{}, 'l_leg':{}}

    def __init__(self, init_pose='walk', dt=0.01, mode='2D', episode_limit=2000, seed=0, slope_deg=0,
                 tgt_field_ver=0, reflex_params=np.ones(73,), target_vel=np.ones(2,)*-1,
                 obs_param=None, rew_type=0, render_mode=None, reward_wt=None,
                 stim_mode='reflex', tgt_vel_mode='eval', sine_vel_args=None, delta_mode='delayed',
                 delta_control_mode='asym', reflex_delay=False):
        """
        init_pose: Initial pose key, read from the XML file.
        mode: 2D or 3D model
        episode_limit: Length of each episode in timesteps
        
        sine_vel_args : A dictionary of paramters to generate the a sine target velocity curve
            min vel, max_vel: Used to calcualte the center velocity of the wave. 

        """

        self.muscle_labels = {}
        self.muscles_dict = {}
        self.muscle_Fmax = {}
        self.muscle_L0 = {}
        self.muscle_LT = {}

        self.seed = seed
        self.mode = mode
        self.init_pose = init_pose

        # Desired slope degree
        # By default, a slope of 0 does not utilize the heightfield
        self.slope_deg = slope_deg

        curr_dir = os.getcwd()
        #curr_dir = '/home/chu.tan/code/Myosuite_2_2_0/myoreflex_workspace/'

        if self.mode == '2D':
            mvt_dim = 2
            pathAndModel = '../models/gait14dof22musc_cvt3_Right_Toeless_2D_RL.xml'
            self.init_state_list = np.load( os.path.join(curr_dir, "save_state_list_2D.npy"), allow_pickle=True)
            self.init_reflex_list = np.load(os.path.join(curr_dir, "save_reflex_list_2D.npy"), allow_pickle=True)
            # print("Using 2D model and state randomization")
        else:
            mvt_dim = 3
            pathAndModel = '../models/gait14dof22musc_cvt3_Right_Toeless_3D.xml'
            # print("Using 3D model")
        self.delayed = reflex_delay
        # !!! IMPT: Set timestep to 0.0005 ms (half a milisec) after env creation below
        self.frame_skip = 10 # Default for Myosuite environments. skip of 10 means each step() is 1 ms (0.01 sec)
        if self.delayed:
            self.frame_skip = 1 # Prepare for 1 ms (0.001 sec) timestep

        self.MyoEnv = gymnasium.make('myoLegStandRandom-v0', 
                            model_path=os.path.join(curr_dir, pathAndModel),
                            normalize_act=False,
                            joint_random_range=(0, 0),
                            frame_skip=self.frame_skip)
        self.MyoEnv.seed(self.seed)
        self.rng_gen = np.random
        self.rng_gen.seed(self.seed)
        # Because we have a 2.5 ms delay for the hip, so the minimum timestep for the underlying simulator has to be 0.5 ms
        # Modify it after creating the environment
        # NOTE: MinDelay modified to 1ms, instead of 0.5ms
        if self.delayed:
            self.MyoEnv.sim.model.opt.timestep = 0.001
            self.MyoEnv.forward()
        
        # Timekeeping
        self.dt = self.MyoEnv.dt
        self.episode_limit = episode_limit
        
        if self.slope_deg != 0:
            self.setupTerrain(self.slope_deg)
       
        # Check if control params are valid and defaults to a valid number.
        if self.mode == '2D' and len(reflex_params) != 47 + 8 + 18 + len(self.mus_len_key):
            print(f"Wrong number of params, Defaulting to {47 + 8 + 18 + len(self.mus_len_key)}")
            reflex_params = np.ones(47 + 8 + 18 + len(self.mus_len_key),)

        elif self.mode == '3D' and len(reflex_params) != 59 + 12 + 22 + len(self.mus_len_key):
            print(f"Wrong number of params, Defaulting to {59 + 12 + 22 + len(self.mus_len_key)}")
            reflex_params = np.ones(59 + 12 + 22 + len(self.mus_len_key),)

        self.CONTROL_PARAM = reflex_params

        if self.mode == '2D':
            init_reflex_param = self.CONTROL_PARAM[0:47]
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[47:len(self.CONTROL_PARAM)])
        elif self.mode == '3D':
            init_reflex_param = self.CONTROL_PARAM[0:59]
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[59:len(self.CONTROL_PARAM)])

        # Store the action space for number of muscles
        self.muscle_space = self.MyoEnv.sim.model.na

        # Fixed variable declarations
        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 0
        self.footstep['l_contact'] = 0

        self._set_muscle_groups()
        self.ReflexCtrl = MyoLocoCtrl_RL(control_dimension=mvt_dim, timestep=self.dt, delayed=self.delayed)
        # self.ReflexCtrl.set_control_params(init_reflex_param) # Only do it in reset

        # Accessor for LocoCtrl
        self.Reflex_cp = self.ReflexCtrl.cp
        self.Reflex_cp_map = self.ReflexCtrl.cp_map
        # self._set_muscle_groups()
        # self.set_init_pose(self.init_pose)

        self.delta_control_mode = delta_control_mode

        self.init_SB_env(obs_param, rew_type, stim_mode, tgt_vel_mode, sine_vel_args, target_vel, delta_mode, reward_wt, tgt_field_ver)

    def setupTerrain(self, slope_degree):
        self.slope_deg = slope_degree # Updating slope in case this function is called from outside of the env
        normalized_data = np.zeros((100,500))

        slope_fill_offset = 449
        slope = np.linspace(0,20,slope_fill_offset) * np.tan(np.deg2rad(np.abs(slope_degree)))

        if slope_degree < 0:
            slope = np.flipud(slope)
            # Update the elevation
            self.MyoEnv.env.sim.model.hfield_size[0,2] = slope[0]
            self.height_offset = slope[0] - 0.005
        else:
            self.MyoEnv.env.sim.model.hfield_size[0,2] = slope[-1]
            self.height_offset = slope[0]

        slope = (slope - slope.min()) / (slope.max() - slope.min())

        normalized_data[:, (500-slope_fill_offset)::] = slope

        if slope_degree < 0:
            normalized_data[:, (500-slope_fill_offset)-3:(500-slope_fill_offset)] = 1

        self.MyoEnv.env.sim.model.hfield_data[:] = normalized_data.reshape(100*500,)
        
        # reinstate heightmap
        self.MyoEnv.env.sim.model.geom_rgba[self.MyoEnv.sim.model.geom_name2id('terrain')][-1] = 1.0
        self.MyoEnv.env.sim.model.geom_pos[self.MyoEnv.sim.model.geom_name2id('terrain')] = np.array([40,0,-0.005])
        self.MyoEnv.env.sim.model.geom_contype[self.MyoEnv.sim.model.geom_name2id('terrain')] = 1
        self.MyoEnv.env.sim.model.geom_conaffinity[self.MyoEnv.sim.model.geom_name2id('terrain')] = 1
        
    def reset(self, seed=None, options=None):

        self.time_step = 0
        self.inner_step = 0

        # Randomizing linear target velocities for training
        #If a target velocity is passed in, then use the input target
        if self.tgt_vel_mode == 'train':
            self.target_vel_randomizer = self.rng_gen.randint(low=1, high=5) # low (inclusive) to high (exclusive)
            self.randomize_target_vel_params()

        if self.tgt_vel_mode == 'constant':
            self.target_vel_type = 'constant'

        self.prev_action = np.zeros(self.action_space.shape)
        
        if seed is not None:
            self.seed = seed

        self.MyoEnv.reset()
        # self.ReflexCtrl.reset()
        self._set_muscle_groups()
        self.set_init_pose(self.init_pose) # Internal calls a forward() after setting initial pose
        
        self.time_step = 0
        self.previous_distance = self.MyoEnv.env.sim.data.body('pelvis').xpos[0].copy()
        # self.vtgt.reset(version=self.tgt_field_ver, seed=self.seed)

        if self.tgt_vel_mode == 'train' and self.rew_type != 1:
            self.randomize_init_state()

        self.reset_control_params()

        body_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        pelvis_euler = self._get_pel_angle()
        
        pose = np.array([body_xpos[0], body_xpos[1], pelvis_euler[2]]) # No negative yaw, local velocity field is internally inverted
        # self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)
        
        if self.rew_type == 1:
            self.obs_dict = self.get_obs_dict_Wang(np.array([0,0,0])) # Initialize velocity to zero
        else:
            self.obs_dict = self.get_obs_dict(np.zeros(self.action_space.shape))

        self.init_reward_1()

        #self.obs_dict = self.get_obs_dict(np.zeros(self.action_space.shape))
        observation = self.get_obs_vec(self.obs_dict.copy())
        info = {}

        self.footstep['new'] = False
        
        # Keep both true to ignore the very first initial step of the initial pose when environment resets
        # update_footstep() will then update normally once a step has been made
        self.footstep['r_contact'] = True
        self.footstep['l_contact'] = True

        # Debug lines
        self.avg_vel = 0
        self.step_vel = 0

        return observation, info

    def step_Wang(self, action):
        """
        Step function as defined in Wang et al (2019)
        Policy updates are only made when there is a new step, not at every time step
        Observations are only made when there is a new step ("policy interacts with environment at the beginning of each bipedal gait cycle")
        """

        self.time_step += 1
        terminated = False
        truncated = False
        #print(f"Timestep - {self.time_step}, Pel Pos: {self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()}")
        
        out_act = action
        if self.delta_control_mode == 'sym':
            ctr_param_R = out_act
            ctr_param_L = out_act
        elif self.delta_control_mode == 'asym':
            ctr_param_R = out_act[0:int(self.action_space.shape[0]/2)] # Python is end index exclusive
            ctr_param_L = out_act[int(self.action_space.shape[0]/2)::]
        
        #print(f"New footstep: {self.footstep['new']}")
        inner_step = 0
        start_pos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()

        # Run simulation until you get a new step
        while not self.footstep['new']: # Used to ignore the first step as initialization
            inner_step += 1
            #print(f"New inner time (In loop): {self.inner_step}, Pel Pos: {self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()}")
            
            self.ReflexCtrl.modulate_reflex_Simplified('r_leg', ctr_param_R)
            self.ReflexCtrl.modulate_reflex_Simplified('l_leg', ctr_param_L)

            self.get_reflex_sensData()
            new_act = self.reflex2mujoco(self.update_reflex_ctr(self.SENSOR_DATA))
            self.MyoEnv.step(new_act)

            self.update_footstep()

            # ----- Checking for ending conditions -----
            body_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
            pelvis_euler = self._get_pel_angle()

            # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
            if body_xpos[2] < 0.75: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
                truncated = True
                break
                #terminated = True
            if pelvis_euler[1] < np.deg2rad(-60) or pelvis_euler[1] > np.deg2rad(60):
                # Punish for too much pitch of pelvis
                truncated = True
                break
                #terminated = True
            
        curr_pos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        current_avg_vel = (curr_pos - start_pos) / (inner_step*self.dt)

        # print(f"Inner time (After)- {inner_step}, Pel Pos: {self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()}")
        # print(f"Dist : {curr_pos[0] - start_pos[0]}, Vel: {(curr_pos[0] - start_pos[0]) / (inner_step*self.dt)}")
        
        # ----- Obs and rewards after all updates -----
        terminated = False # Not used, setting to false. SubProcVec treats terminated and truncated as the same in SB3

        self.obs_dict = self.get_obs_dict_Wang(current_avg_vel)
        observation = self.get_obs_vec(self.obs_dict.copy()) # Added the action as part of observation
        reward = self.get_reward(self.get_reward_dict_Wang(terminated, truncated, action, current_avg_vel))

        # Store memory of current action, as previous action, after reward calculations
        self.prev_action = action.copy()

        info = {'timestep': self.time_step}
        
        # Reset new step for the next step
        self.footstep['new'] = False

        return observation, reward, terminated, truncated, info

    def step(self, action):
        
        terminated = False
        truncated = False
        self.time_step += 1
        """
        RL agent interacts with reflex control parameters, instead of muscles
        """

        #out_act = self.convertRange(action, self.ACT_RANGE, self.REF_GAIN_RANGE)
        out_act = action

        #if self.stim_mode == 'reflex':
            #Split action into half
        if self.delta_control_mode == 'sym':
            ctr_param_R = out_act
            ctr_param_L = out_act
        elif self.delta_control_mode == 'asym':
            ctr_param_R = out_act[0:int(self.action_space.shape[0]/2)] # Python is end index exclusive
            ctr_param_L = out_act[int(self.action_space.shape[0]/2)::]

        if self.delta_mode == 'delayed':
            for delay in range(3): #~30 hz, whose period is 0.033, but step the new params over 3 updates
            
                self.ReflexCtrl.modulate_reflex_gain('r_leg', ctr_param_R)
                self.ReflexCtrl.modulate_reflex_gain('l_leg', ctr_param_L)
                #self.ReflexCtrl.modulate_reflex_targets('r_leg', ctr_param_R)
                #self.ReflexCtrl.modulate_reflex_targets('l_leg', ctr_param_L)
                #self.ReflexCtrl.modulate_reflex_Simplified('r_leg', ctr_param_R)
                #self.ReflexCtrl.modulate_reflex_Simplified('l_leg', ctr_param_L)
                
                # Obtain new activation values from reflex controller and send it into the environment
                self.get_reflex_sensData()
                new_act = self.reflex2mujoco(self.update_reflex_ctr(self.SENSOR_DATA))
                self.MyoEnv.step(new_act)
        else:
            self.ReflexCtrl.modulate_reflex_gain('r_leg', ctr_param_R)
            self.ReflexCtrl.modulate_reflex_gain('l_leg', ctr_param_L)
            #self.ReflexCtrl.modulate_reflex_targets('r_leg', ctr_param_R)
            #self.ReflexCtrl.modulate_reflex_targets('l_leg', ctr_param_L)
            #self.ReflexCtrl.modulate_reflex_Simplified('r_leg', ctr_param_R)
            #self.ReflexCtrl.modulate_reflex_Simplified('l_leg', ctr_param_L)

            # Obtain new activation values from reflex controller and send it into the environment
            self.get_reflex_sensData()
            new_act = self.reflex2mujoco(self.update_reflex_ctr(self.SENSOR_DATA))
            self.MyoEnv.step(new_act)

        self.update_footstep()
        
        # ----- Velocity field updates -----
        body_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        pelvis_euler = self._get_pel_angle()

        pose = np.array([body_xpos[0], body_xpos[1], pelvis_euler[2]]) # No negative yaw, local velocity field is internally inverted
        # self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)

        # ----- Checking for ending conditions -----
        # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
        if self.MyoEnv.env.sim.data.joint('pelvis_ty').qpos[0] < 0.70: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
            truncated = True
            #terminated = True
        if pelvis_euler[1] < np.deg2rad(-60) or pelvis_euler[1] > np.deg2rad(60):
            # Punish for too much pitch of pelvis
            truncated = True
            #terminated = True

        if self.time_step > self.episode_limit-1: # Alive for episode, less than ep limit -1 because time_step is incremented at the beginning of the step
            terminated = True
            # truncated = True

        # ----- Obs and rewards after all updates -----
        self.obs_dict = self.get_obs_dict(action)

        observation = self.get_obs_vec(self.obs_dict.copy()) # Added the action as part of observation
        
        if self.rew_type == 0:
            reward = self.get_reward(self.get_reward_dict_old(terminated, truncated, action))
        elif self.rew_type == 1:
           reward = self.get_reward(self.get_reward_dict_Wang(terminated, truncated, action))
        elif self.rew_type == 2:
            reward = self.get_reward(self.get_reward_dict_posVel(terminated, truncated, action))
        else:
            reward = 0
            print("No reward functions selected, returning 0")

        # Store memory of current action, as previous action, after reward calculations
        self.prev_action = action.copy()
        
        # Reset previous distance covered
        self.previous_distance = self.MyoEnv.env.sim.data.body('pelvis').xpos[0].copy()

        info = {'timestep': self.time_step, 'musc_stim': new_act}

        return observation, reward, terminated, truncated, info

    def get_obs_dict(self, current_action):

        obs_dict = {}
        
        #obs_dict['v_tgt_field'] = np.ndarray.flatten(self.v_tgt_field.copy())
        obs_dict['v_tgt'] = self.get_target_vel()
        #obs_dict['reflex_action'] = current_action.copy()

        obs_dict['pelvis_height'] = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()[2]
        obs_dict['pelvis_euler'] = self._get_pel_angle() # Local body frame Euler angles
        obs_dict['pelvis_euler_vel'] = self._get_pel_angle_vel()#*self.dt # Local angular velocity
        obs_dict['pelvis_xvel'] = self.get_pel_xvel()#*self.dt

        obs_dict['r_leg'] = {}
        obs_dict['l_leg'] = {}

        for s_leg in ['r_leg', 'l_leg']:

            # Joint angles
            obs_dict[s_leg][f"hip_flexion_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos.copy()
            obs_dict[s_leg][f"knee_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos.copy()
            obs_dict[s_leg][f"ankle_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos.copy()

            # Joint velocities
            obs_dict[s_leg][f"vel_hip_flexion_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel.copy()
            obs_dict[s_leg][f"vel_knee_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel.copy()
            obs_dict[s_leg][f"vel_ankle_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qvel.copy()

            if self.mode == '3D':
                obs_dict[s_leg][f"hip_add_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos.copy()
                obs_dict[s_leg][f"vel_hip_add_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qvel.copy()

            temp_GRF = (self.MyoEnv.env.sim.data.sensor(f"{s_leg[0]}_foot").data[0].copy() + self.MyoEnv.env.sim.data.sensor(f"{s_leg[0]}_toes").data[0].copy())
            obs_dict[s_leg][f"GRF_{s_leg[0]}"] = temp_GRF / (np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8)

            #temp_mus_force = self.MyoEnv.env.sim.data.actuator_force.copy()
            #temp_mus_len = self.MyoEnv.env.sim.data.actuator_length.copy()
            #temp_mus_vel = self.MyoEnv.env.sim.data.actuator_velocity.copy()

            #if self.obs_param is not None:
            #    if 'mus_f' in self.obs_param:
            #        for MUS in self.muscles_dict[s_leg].keys():
            #            obs_dict[s_leg][f"{s_leg[0]}_{MUS}_f"] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] / (self.muscle_Fmax[s_leg][MUS]) )

            #if self.obs_param is not None:
            #    if 'mus_l' in self.obs_param:
            #        for MUS in self.muscles_dict[s_leg].keys():
            #            obs_dict[s_leg][f"{s_leg[0]}_{MUS}_l"] = ( temp_mus_len[self.muscles_dict[s_leg][MUS]] - self.muscle_LT[s_leg][MUS] ) / self.muscle_L0[s_leg][MUS]

            #if self.obs_param is not None:
            #    if 'mus_v' in self.obs_param:
            #        for MUS in self.muscles_dict[s_leg].keys():
            #            obs_dict[s_leg][f"{s_leg[0]}_{MUS}_v"] = temp_mus_vel[self.muscles_dict[s_leg][MUS]] / self.muscle_L0[s_leg][MUS]

            # for MUS in self.muscles_dict[s_leg].keys():
            #     obs_dict[s_leg][f"{s_leg[0]}_{MUS}_f"] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] / (self.muscle_Fmax[s_leg][MUS]) )
            #     obs_dict[s_leg][f"{s_leg[0]}_{MUS}_l"] = ( temp_mus_len[self.muscles_dict[s_leg][MUS]] - self.muscle_LT[s_leg][MUS] ) / self.muscle_L0[s_leg][MUS]
            #     obs_dict[s_leg][f"{s_leg[0]}_{MUS}_v"] = temp_mus_vel[self.muscles_dict[s_leg][MUS]] / self.muscle_L0[s_leg][MUS]

        obs_dict['foot_pos'] = self._get_feet_relative_position()
        obs_dict['mus_act'] = self._get_muscle_act()

        # Internal states of reflex controller
        obs_dict['spinal_phase'] = self.ReflexCtrl.get_spinal_phase()
        obs_dict['supraspinal_command'] = self.ReflexCtrl.get_supraspinal_command()

        return obs_dict
    
    def get_obs_dict_Wang(self, current_avg_vel):
        
        obs_dict = {}
        
        obs_dict['v_tgt'] = self.get_target_vel()

        obs_dict['pelvis_height'] = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()[2]
        obs_dict['pelvis_euler'] = self._get_pel_angle() # Local body frame Euler angles
        obs_dict['pelvis_euler_vel'] = self._get_pel_angle_vel()#*self.dt # Local angular velocity
        obs_dict['pelvis_xvel'] = current_avg_vel

        obs_dict['r_leg'] = {}
        obs_dict['l_leg'] = {}

        for s_leg in ['r_leg', 'l_leg']:

            # Joint angles
            obs_dict[s_leg][f"hip_flexion_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos.copy()
            obs_dict[s_leg][f"knee_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos.copy()
            obs_dict[s_leg][f"ankle_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos.copy()

            # Joint velocities
            obs_dict[s_leg][f"vel_hip_flexion_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel.copy()
            obs_dict[s_leg][f"vel_knee_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel.copy()
            obs_dict[s_leg][f"vel_ankle_angle_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qvel.copy()

            if self.mode == '3D':
                obs_dict[s_leg][f"hip_add_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos.copy()
                obs_dict[s_leg][f"vel_hip_add_{s_leg[0]}"] = self.MyoEnv.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qvel.copy()

            temp_GRF = (self.MyoEnv.env.sim.data.sensor(f"{s_leg[0]}_foot").data[0].copy() + self.MyoEnv.env.sim.data.sensor(f"{s_leg[0]}_toes").data[0].copy())
            obs_dict[s_leg][f"GRF_{s_leg[0]}"] = temp_GRF / (np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8)

        obs_dict['foot_pos'] = self._get_feet_relative_position()

        return obs_dict

    def get_obs_vec(self, obs_dict):
        obsvec = np.zeros(0)
        for key in obs_dict.keys():
            #print(f"key : {key}")
            if key == 'r_leg' or key == 'l_leg':
                for subkey in obs_dict[key]:
                    #print(f"key : {key}_{subkey}")
                    obsvec = np.concatenate([obsvec, obs_dict[key][subkey].ravel()])
            else:
                obsvec = np.concatenate([obsvec, obs_dict[key].ravel()]) # ravel helps with images
        return np.array(obsvec, dtype=np.float32)

    """
    Reward functions
    """

    def get_reward_dict_Wang(self, terminated, truncated, current_action, current_avg_vel):

        rew_dict = {}
        
        # Velocity
        #current_vel = self.get_pel_xvel()
        tgt_vel = self.get_target_vel()

        # Reward dict compilation here
        #rew_dict['alive_rew'] = self.d_reward['alive']
        rew_dict['v_tgt'] = np.exp(-1*np.square(tgt_vel[0] - current_avg_vel[0])) # Take only X direction velocity

        #self.debug_reward_dict.append(copy.deepcopy(rew_dict))
        self.debug_reward_dict = copy.deepcopy(rew_dict)
        self.debug_actions = current_action.copy()

        # Reward is zero if the model falls
        if truncated:
            #rew_dict['alive_rew'] = 0
            #rew_dict['footstep'] = 0
            rew_dict['v_tgt'] = 0
            #rew_dict['effort'] = 0
            #rew_dict['terminated'] = 0
            #rew_dict['action_penalty_zero'] = 0
            #rew_dict['action_penalty'] = 0

        return rew_dict

    def get_reward(self, reward_dict):
        
        #reward_dict = self.get_reward_dict(truncated, obs_dict)
        reward = np.sum([wt*reward_dict[key] for key, wt in self.reward_wt.items()], axis=0)
        
        return reward

    def set_reward_weights(self, reward_weights):
        # Make sure it is a dictionary of values
        self.reward_wt = reward_weights

    def get_reward_dict_old(self, terminated, truncated, current_action):
        reward = 0
        dt = 0.01
        rew_dict = {}

        # alive reward
        rew_dict['alive_rew'] = self.d_reward['alive']
        
        # effort ~ muscle fatigue ~ (muscle activation)^2
        ACT2 = 0
        temp_leg = ['r_leg', 'l_leg']
        temp_act = self._get_muscle_act()
        
        for leg in temp_leg:
            for MUS in self.muscles_dict[leg].keys():
                ACT2 += np.sum(np.square(temp_act[self.muscles_dict[leg][MUS]]))
        
        # target velocity
        tgt_vel = self.get_target_vel()
        curr_vel = self.get_pel_xvel()

        rew_dict['footstep'] = 0.1 * self.footstep['new']
        rew_dict['v_tgt'] = np.exp(-20 * np.square(tgt_vel[0] - curr_vel[0]))
        rew_dict['effort'] = np.exp(-1 * ACT2)
        rew_dict['terminated'] = np.int64(terminated)
        rew_dict['truncated'] = np.int64(truncated)

        # action delta penalty
        rew_dict['action_penalty_zero'] = np.exp(-1 * np.mean(np.square(current_action)))
        rew_dict['action_penalty'] = np.exp(-1 * np.mean(np.square(self.prev_action.copy() - current_action)))
        
        # distance reward
        rew_dict['distance'] = self._get_distance_reward(tgt_vel[0], dt)

        #self.debug_reward_dict.append(copy.deepcopy(rew_dict))
        self.debug_reward_dict = copy.deepcopy(rew_dict)
        self.debug_actions = current_action.copy()

        return rew_dict

    def get_reward_dict_posVel(self, terminated, truncated, current_action):

        reward = 0
        dt = 0.01
        rew_dict = {}

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        rew_dict['alive_rew'] = self.d_reward['alive']
        # effort ~ muscle fatigue ~ (muscle activation)^2 
        # Metabolic cost
        ACT2 = 0
        temp_leg = ['r_leg', 'l_leg']
        temp_act = self._get_muscle_act()
        
        for leg in temp_leg:
            for MUS in self.muscles_dict[leg].keys():
                # np.sum used here because there are multiple muscles in each "bundle"
                ACT2 += np.sum(np.square( temp_act[self.muscles_dict[leg][MUS]] ))
        
        # Accumulates from timesteps, no reward if no new step, so zero
        rew_dict['footstep'] = 0
        rew_dict['v_tgt'] = 0
        rew_dict['effort'] = 0
        rew_dict['terminated'] = 0

        # Accumulator
        self.d_reward['footstep']['effort'] += ACT2*dt
        self.d_reward['footstep']['del_t'] += dt
        # reward from velocity (penalize from deviating from v_tgt)
        self.d_reward['footstep']['del_v'] += self._get_vel_diff(self.get_pel_xvel())*dt
        self.step_vel += self.get_pel_xvel()*dt
        self.avg_vel = 0

        # footstep reward (when made a new step)
        if self.footstep['new']:
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            
            if self.d_reward['footstep']['del_t'] == 0:
                self.avg_vel = 0
            else:
                self.avg_vel = (self.step_vel / self.d_reward['footstep']['del_t'])[0]
            
            #avg_diff = self.d_reward['footstep']['del_v'] / self.d_reward['footstep']['del_t']
            avg_diff = self._get_vel_diff(self.avg_vel)
            
            reward_footstep_v = self.d_reward['weight']['v_tgt']*( np.sum( np.exp(-np.square(avg_diff)) ) )

            # penalize effort
            reward_footstep_e = -1*self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

            self.d_reward['footstep']['del_t'] = 0
            self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0

            #reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e
            rew_dict['footstep'] = reward_footstep_0
            rew_dict['v_tgt'] = reward_footstep_v
            rew_dict['effort'] = reward_footstep_e

            self.step_vel = 0

        # success bonus
        if terminated and (truncated == False): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            rew_dict['footstep'] += self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

        # Action delta penalty
        rew_dict['action_penalty_zero'] = -1*np.mean( np.square(current_action) ) #np.mean(np.exp(-1000 * current_action**2)-0.1)
        rew_dict['action_penalty'] = -1*np.mean( np.square(self.prev_action.copy() - current_action) )
        
        #self.debug_reward_dict.append(copy.deepcopy(rew_dict))
        self.debug_reward_dict = copy.deepcopy(rew_dict)
        self.debug_actions = current_action.copy()

        return rew_dict

    """
    Environment and state randomization functions
    """
    def randomize_init_state(self):
        rnd_state_idx = self.rng_gen.randint(low=0, high=self.init_rnd_state_len)

        # Randomize the index based on total number
        selected_state = copy.deepcopy(self.init_state_list[rnd_state_idx])
        selected_reflex = copy.deepcopy(self.init_reflex_list[rnd_state_idx])

        self.set_reflex_env_state(selected_state)
        self.ReflexCtrl = selected_reflex

    def randomize_target_vel_params(self):
        # Randomizing target velocities for training
        #If a target velocity is passed in, then use the input target

        """
        Randomize velocities based on percentage
        """
        self.target_x_vel = np.clip( np.round(self.rng_gen.uniform(low=0.8, high=1.81), 2), 0.8, 1.8)
        self.target_y_vel = 0
        self.target_vel_type = 'constant'
        
        #if self.target_vel_randomizer in np.array([1,2]):
        #    self.target_x_vel = np.clip( np.round(self.rng_gen.uniform(low=0.8, high=1.81), 2), 0.8, 1.8)
        #    self.target_y_vel = 0
        #    self.target_vel_type = 'constant'
        
        """
        elif self.target_vel_randomizer == 3:
            diff = self.rng_gen.uniform(low=0.1, high=0.81)
            min_tgt = self.rng_gen.uniform(low=0.8, high=1.0)
            sorted_tgt = np.array([min_tgt, min_tgt+diff])

            if self.rng_gen.randint(low=0, high=2) == 0:
                sorted_tgt = sorted_tgt[::-1]

            self.single_change_vel = sorted_tgt
            self.single_change_time = self.rng_gen.randint(low=5, high=15) * 100

            self.target_vel_type = 'constant_change'

        elif self.target_vel_randomizer == 4:
            rnd_vel = np.sort(np.clip( np.round(self.rng_gen.uniform(low=0.8, high=1.81, size=2), 2), 0.8, 1.8))
            self.sine_min = rnd_vel[0]
            self.sine_max = rnd_vel[1]
            self.sine_period = self.rng_gen.randint(low=5, high=21) * 100 # simulation timestep is 0.01 sec.
            self.phase_shift = self.rng_gen.randint(low=0, high=11) * 100
            self.target_vel_type == 'sine'
        """

        #print(self.target_vel_type)

    def get_target_vel(self):
        #world_com_xpos = self.MyoEnv.sim.data.body('pelvis').xpos.copy()
        #v_tgt = self.vtgt.get_vtgt(world_com_xpos[0:2]).T

        if self.tgt_vel_mode == 'eval':
            return np.array([self.eval_x_vel, self.eval_y_vel])
            
        elif self.target_vel_type == 'constant':
            return np.array([self.target_x_vel, self.target_y_vel])

        elif self.target_vel_type == 'sine':
            return self.get_sinusoidal_vel(self.time_step)

        elif self.target_vel_type == 'constant_change':

            target_x_vel = self.single_change_vel[0]
            if self.time_step == self.single_change_time:
                target_x_vel = self.single_change_vel[1]

            return np.array([target_x_vel, 0])

    def get_sinusoidal_vel(self, current_time):
        """
        Compute the value of a sine wave at a specific time.
        Current time: Given in milliseconds
        """
        #phase_shift = 0
        
        amplitude = (self.sine_max - self.sine_min) / 2
        offset = (self.sine_min + self.sine_max) / 2

        frequency = 1 / self.sine_period
        value = amplitude * np.sin(2 * np.pi * frequency * current_time + self.phase_shift) + offset

        return np.array([value, 0]) # Currently olny for 2D walking

    """
    Utility functions
    """

    def _get_vel_diff(self, current_pel_vel):
        
        vel_diff = (current_pel_vel - self.get_target_vel())

        return vel_diff

    def _get_vel_reward(self, current_pel_vel):
    
        #vel_diff = self._get_vel_diff(current_pel_vel)
        tgt_vel = self.get_target_vel()
        return np.exp(-np.square(tgt_vel[0] - current_pel_vel[0])) + np.exp(-np.square(tgt_vel[1] - current_pel_vel[1]))

    def _get_pos_reward(self):
        """
        Reward for how close agent is to the goal
        """
        pel_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        # np.exp(-np.square(self.target_y_pose - pel_xpos[1])) + np.exp(-np.square(self.target_x_pose - pel_xpos[0]))
        return np.exp(-np.square(self.target_x_pose - pel_xpos[0]))

    def _get_distance_reward(self, target_velocity, dt):

        current_distance = self.MyoEnv.env.sim.data.body('pelvis').xpos[0].copy()
        distance_covered = current_distance - self.previous_distance
        expected_distance = target_velocity * dt


        if expected_distance > 0:
            return np.exp(-500 * np.square(expected_distance - distance_covered))
        else:
            return 0

    def init_reward_1(self):
        self.d_reward = {}

        self.d_reward['weight'] = {}
        self.d_reward['weight']['footstep'] = 10
        self.d_reward['weight']['effort'] = 1
        self.d_reward['weight']['v_tgt'] = 1
        self.d_reward['weight']['v_tgt_R2'] = 3

        self.d_reward['alive'] = 0.1 #0.21 # Increased it from 0.1 due to addition of action penalty
        self.d_reward['effort'] = 0

        self.d_reward['footstep'] = {}
        self.d_reward['footstep']['effort'] = 0
        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0

    """
    Reflex Controller functions
    """
    # Run without modulating inputs
    def run_reflex_step(self):
        # Run a step of the Mujoco env and Reflex controller
        is_done = False
        
        self.get_reflex_sensData()
        new_act = self.reflex2mujoco(self.update_reflex_ctr(self.SENSOR_DATA))

        self.MyoEnv.step(new_act)
        
        self.update_footstep()

        body_xquat = self.MyoEnv.env.sim.data.body('pelvis').xquat.copy()
        world_com_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)

        # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
        if world_com_xpos[2] < 0.65: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
            is_done = True
        if pelvis_euler[1] < np.deg2rad(-60) or pelvis_euler[1] > np.deg2rad(60):
            # Punish for too much pitch of pelvis
            is_done = True

        return new_act, is_done

    def update_reflex_ctr(self, sens_dict):
        return self.ReflexCtrl.update(sens_dict)
    
    def get_reflex_sensData(self):

        # Calculating intrinsic Euler angles (in body frame)
        body_xquat = self.MyoEnv.sim.data.body('pelvis').xquat.copy()
        world_com_xpos = self.MyoEnv.sim.data.body('pelvis').xpos.copy()
        world_com_xvel = self.MyoEnv.sim.data.object_velocity('pelvis','body', local_frame=False)[0].copy()

        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        pelvis_euler_vel = self.MyoEnv.sim.data.object_velocity('pelvis', 'body', local_frame=True).copy()

        self.SENSOR_DATA['body']['theta'] = pelvis_euler[1] # Forward tilt (+) after conversion
        self.SENSOR_DATA['body']['dtheta'] = -1*pelvis_euler_vel[1][2] # velocity about z-axis (z-axis points to the right of the model), forward (+)
        self.SENSOR_DATA['body']['theta_f'] = pelvis_euler[0] - np.deg2rad(90) # Right roll (+), Left list (-)
        self.SENSOR_DATA['body']['dtheta_f'] = pelvis_euler_vel[1][0] # Right roll (+)
        
        # Calculating sagittal plane local coordinates
        x_local, y_local = self._rotate_frame(world_com_xpos[0], world_com_xpos[1], -1*pelvis_euler[2]) # Yaw, Left (+) Right (-) 
        dx_local, dy_local = self._rotate_frame(world_com_xvel[0], world_com_xvel[1], -1*pelvis_euler[2])

        self.SENSOR_DATA['body']['pelvis_pos'] = np.array([x_local, y_local]) # Local coord (+ direction) [(Forward), (Leftward)]
        self.SENSOR_DATA['body']['pelvis_vel'] = np.array([dx_local, dy_local]) # Local coord (+ direction) [(Forward), (Leftward)]

        # GRF from foot contact sensor values
        temp_right = (self.MyoEnv.sim.data.sensor('r_foot').data[0].copy() + self.MyoEnv.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.MyoEnv.sim.data.sensor('l_foot').data[0].copy() + self.MyoEnv.sim.data.sensor('l_toes').data[0].copy())

        self.SENSOR_DATA['r_leg']['load_ipsi'] = temp_right / (np.sum(self.MyoEnv.sim.model.body_mass)*9.8)
        self.SENSOR_DATA['l_leg']['load_ipsi'] = temp_left / (np.sum(self.MyoEnv.sim.model.body_mass)*9.8)

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            # GRF data for spinal phrases
            self.SENSOR_DATA[s_leg]['contact_ipsi'] = 1 if self.SENSOR_DATA[s_leg]['load_ipsi'] > 0.1 else 0
            self.SENSOR_DATA[s_leg]['contact_contra'] = 1 if self.SENSOR_DATA[s_legc]['load_ipsi'] > 0.1 else 0
            self.SENSOR_DATA[s_leg]['load_contra'] = self.SENSOR_DATA[s_legc]['load_ipsi']

            tal_world_xpos = self.MyoEnv.sim.data.body(f"talus_{s_legc[0]}").xpos.copy()
            tal_world_xvel = self.MyoEnv.sim.data.object_velocity(f"talus_{s_legc[0]}",'body', local_frame=False)[0].copy()

            tal_x_local, tal_y_local = self._rotate_frame(tal_world_xpos[0], tal_world_xpos[1], -1*pelvis_euler[2])
            tal_dx_local, tal_dy_local = self._rotate_frame(tal_world_xvel[0], tal_world_xvel[1], -1*pelvis_euler[2])
            
            # Alpha tgt calculations
            self.SENSOR_DATA[s_leg][f"talus_contra_pos"] = np.array([tal_x_local, tal_y_local])
            self.SENSOR_DATA[s_leg][f"talus_contra_vel"] = np.array([tal_dx_local, tal_dy_local])
            # object_velocity from DMcontrol - {https://github.com/deepmind/dm_control/blob/d6f9cb4e4a616d1e1d3bd8944bc89541434f1d49/dm_control/mujoco/wrapper/core.py#L481}

            # Leg joint angles
            self.SENSOR_DATA[s_leg]['phi_hip'] = (np.pi - self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            self.SENSOR_DATA[s_leg]['phi_knee'] = (np.pi + self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            self.SENSOR_DATA[s_leg]['phi_ankle'] = (0.5*np.pi - self.MyoEnv.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0].copy())

            self.SENSOR_DATA[s_leg]['dphi_hip'] = -1*self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel[0].copy() # Alpha calculation conversions
            self.SENSOR_DATA[s_leg]['dphi_knee'] = self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel[0].copy() # Alpha calculation conversions

            # Check sign - BODY FRAME ALPHA
            # self.SENSOR_DATA[s_leg]['alpha'] = self.SENSOR_DATA[s_leg]['phi_hip'] - 0.5*self.SENSOR_DATA[s_leg]['phi_knee'] 
            # self.SENSOR_DATA[s_leg]['dalpha'] = -1*self.SENSOR_DATA[s_leg]['dphi_hip'] - 0.5*self.SENSOR_DATA[s_leg]['dphi_knee'] # Hip flexion vel (-), Knee flexion vel (-)  Only for dalpha calculations
            if self.mode == '3D':
                self.SENSOR_DATA[s_leg]['phi_hip_add'] = (self.MyoEnv.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy() + 0.5*np.pi) # Inwards (Add, +), Outwards (Abd, -), for alpha_f
                self.SENSOR_DATA[s_leg]['phi_hip_rot'] = (self.MyoEnv.sim.data.joint(f"hip_rotation_{s_leg[0]}").qpos[0].copy()) # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot
                self.SENSOR_DATA[s_leg]['dphi_hip_rot'] = (self.MyoEnv.sim.data.joint(f"hip_rotation_{s_leg[0]}").qvel[0].copy()) # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot

            temp_mus_force = self.MyoEnv.sim.data.actuator_force.copy()
            #temp_mus_len = self.MyoEnv.sim.data.actuator_length.copy()
            #temp_mus_vel = self.MyoEnv.sim.data.actuator_velocity.copy()

            self.SENSOR_DATA[s_leg]['F_GLU'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['GLU']] / (self.muscle_Fmax[s_leg]['GLU']) )
            self.SENSOR_DATA[s_leg]['F_VAS'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['VAS']] / (self.muscle_Fmax[s_leg]['VAS']) )
            self.SENSOR_DATA[s_leg]['F_SOL'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['SOL']] / (self.muscle_Fmax[s_leg]['SOL']) )
            self.SENSOR_DATA[s_leg]['F_GAS'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['GAS']] / (self.muscle_Fmax[s_leg]['GAS']) )
            self.SENSOR_DATA[s_leg]['F_HAM'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['HAM']] / (self.muscle_Fmax[s_leg]['HAM']) )
            self.SENSOR_DATA[s_leg]['F_HAB'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['HAB']] / (self.muscle_Fmax[s_leg]['HAB']) )

    def _set_model_init_stat(self):
        self.set_init_pose(key_name=self.init_pose)
        self.adjust_initial_pose_cmaes()
        self.adjust_model_height()

    def reset_control_params(self):
        
        if self.mode == '2D':
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[47:len(self.CONTROL_PARAM)])
        elif self.mode == '3D':
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[59:len(self.CONTROL_PARAM)])

        # Setting pose automatically again, based on the 
        if self.mode == '2D':
            #flag_ctrl_mode = '2D' # use 2D
            #param_num = 47 + 9 # 9 more for pose optimization
            if len(self.CONTROL_PARAM) != 47 + 8 + 18: #+ len(self.mus_len_key):
                print(f"Wrong number of params, should be 47 + 8 + 18")
            reflex_params = self.CONTROL_PARAM[0:47]

        elif self.mode == '3D':
            #flag_ctrl_mode = '3D'
            #param_num = 59 + 13 # 13 more for pose optimization
            if len(self.CONTROL_PARAM) != 59 + 12 + 22: #+ len(self.mus_len_key):
                print(f"Wrong number of params, should be 59 + 16 + 22")
            reflex_params = self.CONTROL_PARAM[0:59]

        if self.delayed:
            # Make one single timestep to obtain the joint velocities after reset.
            # Mainly for delayed version, since joint velocities are also used in reflex module output calculations
        #    self.init_musc_state()
            self.MyoEnv.step(np.zeros(self.action_space,))

        # Reset reflex controller with new data after pose has been set
        # Allows the correct initial values to be updated into the controller
        self.get_reflex_sensData()
        #if self.delayed:
        self.ReflexCtrl.reset_spinal_phases(self.init_pose)
        self.ReflexCtrl.reset_delay_buffers(self.SENSOR_DATA, self.init_pose, self.DEFAULT_INIT_MUSC) # , updateFlag=True
        self.ReflexCtrl.reset(reflex_params)
        # Update initial state of the higher layer spinal control
        self.ReflexCtrl.update_supraspinal_control()

    def update_footstep(self):

        # Getting only the heel contacts for new step detection
        r_contact = True if (self.MyoEnv.env.sim.data.sensor('r_foot').data[0].copy()) > 0.1*(np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8) else False
        l_contact = True if (self.MyoEnv.env.sim.data.sensor('l_foot').data[0].copy()) > 0.1*(np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8) else False

        self.footstep['new'] = False
        if ( (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact) ):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def reflex2mujoco(self, output):
        
        mus_act = np.zeros((self.muscle_space,))
        mus_act[:] = 0 # Set unused muscles to 0.

        if self.mode == '3D':
            temp_mus = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
        else:
            temp_mus = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']

        legs = ['r_leg', 'l_leg']
        #musc_idx = self.muscles_dict['r_leg'].keys()

        for s_leg in legs:
            for musc in temp_mus:
                mus_act[self.muscles_dict[s_leg][musc]] = output[s_leg][musc]
        
        return mus_act

    """
    Initialization functions
    """


    def set_init_pose(self, key_name='stand'):
        self.MyoEnv.env.sim.data.qpos = self.MyoEnv.env.sim.model.keyframe(key_name).qpos
        self.MyoEnv.env.sim.data.qvel = self.MyoEnv.env.sim.model.keyframe(key_name).qvel
        self.MyoEnv.env.forward()

    def adjust_initial_pose(self, joint_dict):
        """
        Function allows for additional adjustment of the joint angles from the pre-defined named poses
        """
        # Values in radians
        for joint_name in joint_dict['joint_angles'].keys():
            self.MyoEnv.env.sim.data.joint(joint_name).qpos[0] = joint_dict['joint_angles'][joint_name]

        self.MyoEnv.env.forward()

    def get_pose_cmaes(self, jnt_params):

        pose_dict = {}
        pose_dict['pelvis_ty'] = jnt_params[self.pose_map['pelvis_ty']] *0.01 + 0.868 # *0.1 + 0.778
        pose_dict['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180) # *2*np.pi/180 + (-17*np.pi/180)
        pose_dict['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
        pose_dict['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
        pose_dict['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
        pose_dict['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-5*np.pi/180)
        pose_dict['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
        pose_dict['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-15*np.pi/180)

        if self.mode =='3D':
            pose_dict['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
            pose_dict['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
            pose_dict['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
            pose_dict['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

        pose_dict['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4 #*0.2 + 1.3

        return pose_dict
    
    def update_init_pose_param_cmaes(self,jnt_params):
        # Pelvis tilt, height (pelvis_ty)
        # hip, knee, ankle
        # forward velocity
        if self.mode =='2D' and len(jnt_params) != 8 + 18: # 36
            raise Exception(f'2D mode: Wrong number of pose params. Should be {8 + 18}')
        
        if self.mode =='3D' and len(jnt_params) != 12 + 22 + len(self.mus_len_key): # 46
            raise Exception(f'3D mode: Wrong number of pose params. Should be {12  + 22 + len(self.mus_len_key)}')

        # mus_len_param = jnt_params[-len(self.mus_len_key)::]
    
        # for mus_key in self.mus_len_key:
        #     self.MUS_LENRANGE[mus_key] = mus_len_param[self.mus_len_map[mus_key]]

        # Adjusted such that joint angles add up to the initial defined pose
        # Angles are in the Mujoco convention
        
        self.DEFAULT_INIT_MUSC[self.mode] = {}

        self.JNT_OPTIM['joint_angles'] = {}
        # self.JNT_OPTIM['joint_angles']['pelvis_ty'] = jnt_params[self.pose_map['pelvis_ty']] *0.01 + 0.868 + self.height_offset # *0.1 + 0.778
        self.JNT_OPTIM['joint_angles']['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180) # *2*np.pi/180 + (-17*np.pi/180)
        self.JNT_OPTIM['joint_angles']['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
        self.JNT_OPTIM['joint_angles']['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
        self.JNT_OPTIM['joint_angles']['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
        self.JNT_OPTIM['joint_angles']['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-5*np.pi/180)
        self.JNT_OPTIM['joint_angles']['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
        self.JNT_OPTIM['joint_angles']['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-10*np.pi/180)

        if self.mode =='3D':
            self.JNT_OPTIM['joint_angles']['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

        self.JNT_OPTIM['model_vel'] = {}
        self.JNT_OPTIM['model_vel']['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4 #*0.2 + 1.3

        if self.mode == '2D':
            # Last 30 is for acts
            act_params = jnt_params[8:8+18]

            self.DEFAULT_INIT_MUSC[self.mode]['r_leg'] = {}
            self.DEFAULT_INIT_MUSC[self.mode]['l_leg'] = {}

            for musc in ['GLU_r', 'HFL_r', 'HAM_r', 'RF_r', 'BFSH_r', 'GAS_r', 'SOL_r', 'VAS_r', 'TA_r', 
                    'GLU_l', 'HFL_l', 'HAM_l', 'RF_l', 'BFSH_l', 'GAS_l', 'SOL_l', 'VAS_l', 'TA_l']:
                
                self.DEFAULT_INIT_MUSC[self.mode][f"{musc[-1]}_leg"][f"{musc[0:-2]}"] = act_params[self.init_act_map[musc]] * 0.01
        
        elif self.mode == '3D':
            # Last 30 is for acts
            act_params = jnt_params[12:12+22]

            self.DEFAULT_INIT_MUSC[self.mode]['r_leg'] = {}
            self.DEFAULT_INIT_MUSC[self.mode]['l_leg'] = {}

            for musc in self.init_act_map.keys():
                self.DEFAULT_INIT_MUSC[self.mode][f"{musc[-1]}_leg"][f"{musc[0:-2]}"] = act_params[self.init_act_map[musc]] * 0.01

    def adjust_initial_pose_cmaes(self):

        # Values in radians
        for joint_name in self.JNT_OPTIM['joint_angles'].keys():
            self.MyoEnv.sim.data.joint(joint_name).qpos[0] = self.JNT_OPTIM['joint_angles'][joint_name]
        
        for vel in self.JNT_OPTIM['model_vel'].keys():
            tmp_var = vel.split('_')
            self.MyoEnv.sim.data.joint(f"{tmp_var[1]}_{tmp_var[2]}").qvel[0] = self.JNT_OPTIM['model_vel'][vel]

        self.MyoEnv.sim.data.act[:] = 0.01

        for leg in ['r_leg', 'l_leg']:
            for musc in self.DEFAULT_INIT_MUSC[self.mode][leg].keys():
                self.MyoEnv.sim.data.act[self.muscles_dict[leg][musc]] = self.DEFAULT_INIT_MUSC[self.mode][leg][musc]        
        
        # Scale muscle lengthrange
        # for mus_key in self.MUS_LENRANGE.keys():
        #     self.MyoEnv.sim.model.actuator(f"{mus_key[0:-2]}_l").lengthrange[np.int64(mus_key[-1])] = self.MyoEnv.sim.model.actuator(f"{mus_key[0:-2]}_l").lengthrange[np.int64(mus_key[-1])] * self.MUS_LENRANGE[mus_key]
        #     self.MyoEnv.sim.model.actuator(f"{mus_key[0:-2]}_r").lengthrange[np.int64(mus_key[-1])] = self.MyoEnv.sim.model.actuator(f"{mus_key[0:-2]}_r").lengthrange[np.int64(mus_key[-1])] * self.MUS_LENRANGE[mus_key]


        # Run forward() after modifying and joint angles or velocities
        self.MyoEnv.forward()

    def adjust_model_height(self):
        temp_sens_height = 100
        for sens_site in ['r_heel_btm', 'r_toe_btm', 'l_heel_btm', 'l_toe_btm']:
            if temp_sens_height > self.MyoEnv.sim.data.site(sens_site).xpos[2]:
                temp_sens_height = self.MyoEnv.sim.data.site(sens_site).xpos[2].copy()

        diff_height = self.height_offset - temp_sens_height # Small offset -0.0105
        if self.mode == '2D':
            self.MyoEnv.sim.data.joint('pelvis_ty').qpos[0] = self.MyoEnv.sim.data.joint('pelvis_ty').qpos[0] + diff_height
        else:
            self.MyoEnv.sim.data.joint('pelvis_ty').qpos[0] = self.MyoEnv.sim.data.joint('pelvis_ty').qpos[0] + diff_height
        
        self.MyoEnv.forward()



    def check_pose_validity(self):
        """
        Function to check for if the pose is valid
        """
        is_valid = True

        # Check joint limits
        joints_vec = ['hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']
        if self.mode == '3D':
            joints_vec = ['hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l', 
                          'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l']
            
        for jnts in joints_vec:
            #print(f"Joint limits: {self.MyoEnv.env.sim.model.joint(jnts).range}, IsWithinRange: {self.MyoEnv.sim.data.joint(jnts).qpos[0].copy()}")
            if not (self.MyoEnv.env.sim.data.joint(jnts).qpos[0].copy() >= self.MyoEnv.env.sim.model.joint(jnts).range[0] and self.MyoEnv.env.sim.data.joint(jnts).qpos[0].copy() <= self.MyoEnv.env.sim.model.joint(jnts).range[1]):
                is_valid = False
                return is_valid

        pelvis_euler = self._get_pel_angle()

        if pelvis_euler[1] < np.deg2rad(-30) or pelvis_euler[1] > np.deg2rad(30):
            # Punish for too much pitch of pelvis
            is_valid = False
            return is_valid

        # First 2 indices are world and ground, start from 2
        if np.any(self.MyoEnv.sim.data.xpos[2:][:,2] < 0.005): #Threhold set at 0.2 of body weight, from foot sensors
            is_valid = False
            return is_valid

        # Ensure at least 1 foot is on the ground
        # Foot sensor positions have negative height, so not that informative to check them. Checking the vertical GRF is better
        foot_sens = ['l_foot', 'r_foot', 'l_toes', 'r_toes']
        grf_values = []
        for sens in foot_sens:
            grf_values.append( self.MyoEnv.sim.data.sensor(sens).data[0] / (np.sum(self.MyoEnv.sim.model.body_mass)*9.8) )
        
        if not np.any(np.array(grf_values) > 0.1):
            is_valid = False
            return is_valid

        return is_valid
        
    def _set_muscle_groups(self):
        # ----- Gluteus group -----
        glu_r = [self.MyoEnv.sim.model.actuator('glut_max_r').id]

        glu_l = [self.MyoEnv.sim.model.actuator('glut_max_l').id]

        glu_r_lbl = ['glut_max_r']
        glu_l_lbl = ['glut_max_l']

        # ----- Hamstring (semitendinosus and semimembranosus) -----
        ham_r = [self.MyoEnv.sim.model.actuator('hamstrings_r').id]

        ham_l = [self.MyoEnv.sim.model.actuator('hamstrings_l').id]

        ham_r_lbl = ['hamstrings_r']
        ham_l_lbl = ['hamstrings_l']

        # ----- BF short head (biceps femoris) -----
        bfsh_r = [self.MyoEnv.sim.model.actuator('bifemsh_r').id]

        bfsh_l = [self.MyoEnv.sim.model.actuator('bifemsh_l').id]

        bfsh_r_lbl = ['bifemsh_r']
        bfsh_l_lbl = ['bifemsh_l']

        # ----- Gastrocnemius -----
        gas_r = [self.MyoEnv.sim.model.actuator('gastroc_r').id]

        gas_l = [self.MyoEnv.sim.model.actuator('gastroc_l').id]

        gas_r_lbl = ['gastroc_r']
        gas_l_lbl = ['gastroc_l']

        # ----- Soleus -----
        sol_r = [self.MyoEnv.sim.model.actuator('soleus_r').id]

        sol_l = [self.MyoEnv.sim.model.actuator('soleus_l').id]

        sol_r_lbl = ['soleus_r']
        sol_l_lbl = ['soleus_l']

        # ----- Hip Flexors (psoas and iliacus) -----
        hfl_r = [self.MyoEnv.sim.model.actuator('iliopsoas_r').id]

        hfl_l = [self.MyoEnv.sim.model.actuator('iliopsoas_l').id]

        hfl_r_lbl = ['iliopsoas_r']
        hfl_l_lbl = ['iliopsoas_l']

        # ----- Hip Abductors (piriformis, satorius and tensor fasciae latae) -----
        hab_r = [self.MyoEnv.sim.model.actuator('abd_r').id]

        hab_l = [self.MyoEnv.sim.model.actuator('abd_l').id]

        hab_r_lbl = ['abd_r']
        hab_l_lbl = ['abd_l']

        # ----- Hip Adductors (adductor [brevis, longus, magnus], gracilis) -----
        had_r = [self.MyoEnv.sim.model.actuator('add_r').id]

        had_l = [self.MyoEnv.sim.model.actuator('add_l').id]

        had_r_lbl = ['add_r']
        had_l_lbl = ['add_l']

        # ----- rectus femoris -----
        rf_r = [self.MyoEnv.sim.model.actuator('rect_fem_r').id]

        rf_l = [self.MyoEnv.sim.model.actuator('rect_fem_l').id]

        rf_r_lbl = ['rect_fem_r']
        rf_l_lbl = ['rect_fem_l']

        # ----- Vastius group -----
        vas_r = [self.MyoEnv.sim.model.actuator('vasti_r').id]

        vas_l = [self.MyoEnv.sim.model.actuator('vasti_l').id]

        vas_r_lbl = ['vasti_r']
        vas_l_lbl = ['vasti_l']

        # ----- tibialis anterior -----
        ta_r = [self.MyoEnv.sim.model.actuator('tib_ant_r').id]

        ta_l = [self.MyoEnv.sim.model.actuator('tib_ant_l').id]

        ta_r_lbl = ['tib_ant_r']
        ta_l_lbl = ['tib_ant_l']

        # ----- Consolidating into a single dict -----
        self.muscles_dict['r_leg'] = {}
        self.muscles_dict['r_leg']['HAB'] = hab_r
        self.muscles_dict['r_leg']['HAD'] = had_r
        self.muscles_dict['r_leg']['GLU'] = glu_r
        self.muscles_dict['r_leg']['HAM'] = ham_r
        self.muscles_dict['r_leg']['BFSH'] = bfsh_r
        self.muscles_dict['r_leg']['GAS'] = gas_r
        self.muscles_dict['r_leg']['SOL'] = sol_r
        self.muscles_dict['r_leg']['HFL'] = hfl_r
        self.muscles_dict['r_leg']['RF'] = rf_r
        self.muscles_dict['r_leg']['VAS'] = vas_r
        self.muscles_dict['r_leg']['TA'] = ta_r

        self.muscles_dict['l_leg'] = {}
        self.muscles_dict['l_leg']['HAB'] = hab_l
        self.muscles_dict['l_leg']['HAD'] = had_l
        self.muscles_dict['l_leg']['GLU'] = glu_l
        self.muscles_dict['l_leg']['HAM'] = ham_l
        self.muscles_dict['l_leg']['BFSH'] = bfsh_l
        self.muscles_dict['l_leg']['GAS'] = gas_l
        self.muscles_dict['l_leg']['SOL'] = sol_l
        self.muscles_dict['l_leg']['HFL'] = hfl_l
        self.muscles_dict['l_leg']['RF'] = rf_l
        self.muscles_dict['l_leg']['VAS'] = vas_l
        self.muscles_dict['l_leg']['TA'] = ta_l

        # Muscle labels
        self.muscle_labels['r_leg'] = {}
        self.muscle_labels['r_leg']['HAB'] = hab_r_lbl
        self.muscle_labels['r_leg']['HAD'] = had_r_lbl
        self.muscle_labels['r_leg']['GLU'] = glu_r_lbl
        self.muscle_labels['r_leg']['HAM'] = ham_r_lbl
        self.muscle_labels['r_leg']['BFSH'] = bfsh_r_lbl
        self.muscle_labels['r_leg']['GAS'] = gas_r_lbl
        self.muscle_labels['r_leg']['SOL'] = sol_r_lbl
        self.muscle_labels['r_leg']['HFL'] = hfl_r_lbl
        self.muscle_labels['r_leg']['RF'] = rf_r_lbl
        self.muscle_labels['r_leg']['VAS'] = vas_r_lbl
        self.muscle_labels['r_leg']['TA'] = ta_r_lbl

        self.muscle_labels['l_leg'] = {}
        self.muscle_labels['l_leg']['HAB'] = hab_l_lbl
        self.muscle_labels['l_leg']['HAD'] = had_l_lbl
        self.muscle_labels['l_leg']['GLU'] = glu_l_lbl
        self.muscle_labels['l_leg']['HAM'] = ham_l_lbl
        self.muscle_labels['l_leg']['BFSH'] = bfsh_l_lbl
        self.muscle_labels['l_leg']['GAS'] = gas_l_lbl
        self.muscle_labels['l_leg']['SOL'] = sol_l_lbl
        self.muscle_labels['l_leg']['HFL'] = hfl_l_lbl
        self.muscle_labels['l_leg']['RF'] = rf_l_lbl
        self.muscle_labels['l_leg']['VAS'] = vas_l_lbl
        self.muscle_labels['l_leg']['TA'] = ta_l_lbl

        # --- Muscle normalizations ---

        # L0 calculations (https://github.com/deepmind/mujoco/issues/216)
        temp_L0 = (self.MyoEnv.sim.model.actuator_lengthrange[:,1] - self.MyoEnv.sim.model.actuator_lengthrange[:,0]) / (self.MyoEnv.sim.model.actuator_gainprm[:,1] - self.MyoEnv.sim.model.actuator_gainprm[:,0])
        temp_LT = self.MyoEnv.sim.model.actuator_lengthrange[:,0] - (self.MyoEnv.sim.model.actuator_gainprm[:,0] * temp_L0)

        for leg in self.muscles_dict:
            self.muscle_Fmax[leg] = {}
            self.muscle_L0[leg] = {}
            self.muscle_LT[leg] = {}
            for musc in self.muscles_dict[leg]:
                self.muscle_Fmax[leg][musc] = self.MyoEnv.sim.model.actuator_gainprm[self.muscles_dict[leg][musc], 2].copy()
                self.muscle_L0[leg][musc] = temp_L0[self.muscles_dict[leg][musc]]
                self.muscle_LT[leg][musc] = temp_LT[self.muscles_dict[leg][musc]]

    # ----- Misc functions -----
    def get_joint_names(self):
        '''
        Return a list of joint names according to the index ID of the joint angles
        '''
        return [self.MyoEnv.sim.model.joint(joint_id).name for joint_id in range(0, self.MyoEnv.sim.model.njnt)]
    
    def get_actuator_names(self):
        '''
        Return a list of actuator names according to the index ID of the actuators
        '''
        return [self.MyoEnv.sim.model.actuator(act_id).name for act_id in range(0, self.MyoEnv.sim.model.na)]

    def _get_muscle_act(self):
        return self.MyoEnv.env.sim.data.act.copy()

    def _get_pel_angle(self):
        body_xquat = self.MyoEnv.sim.data.body('pelvis').xquat.copy()
        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)

        # Roll: Upright is 0, Right list (+), Left list (-)
        # Pitch: Forward tilt (+) after conversion
        # Yaw: 0 facing right, Left turn (+)
        
        # Roll, pitch, yaw format
        return np.array([pelvis_euler[0] - np.deg2rad(90), pelvis_euler[1], pelvis_euler[2]]) 

    def _get_pel_angle_vel(self):

        pelvis_euler_vel = self.MyoEnv.sim.data.object_velocity('pelvis', 'body', local_frame=True).copy()

        # Roll vel: Right roll (+)
        # Pitch vel: Forward tilt (+)
        # Yaw vel: Left turn (+)

        # Roll, pitch, yaw angular velocity
        return np.array([pelvis_euler_vel[1][0], -1*pelvis_euler_vel[1][2], pelvis_euler_vel[1][1]])

    def get_pel_xvel(self):

        body_xquat = self.MyoEnv.sim.data.body('pelvis').xquat.copy()
        world_com_xvel = self.MyoEnv.sim.data.object_velocity('pelvis','body', local_frame=False)[0].copy()

        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        
        # Calculating sagittal plane local coordinates
        dx_local, dy_local = self._rotate_frame(world_com_xvel[0], world_com_xvel[1], -1*pelvis_euler[2])
        return np.array([dx_local, dy_local])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        # foot_id_l = self.MyoEnv.env.sim.model.body_name2id('talus_l')
        # foot_id_r = self.MyoEnv.env.sim.model.body_name2id('talus_r')
        pelvis = self.MyoEnv.env.sim.data.body('pelvis').xpos[[0,2]].copy()

        # Using sensor site, in place of ball of foot and toe (end of toe in 11 muscle's case, since toes are missing)
        r_foot = self.MyoEnv.env.sim.data.site('r_foot_touch').xpos[[0,2]].copy() - pelvis
        r_toe = self.MyoEnv.env.sim.data.site('r_toes_touch').xpos[[0,2]].copy() - pelvis

        r_ankle = self.MyoEnv.env.sim.data.joint('ankle_angle_r').xanchor[[0,2]].copy() - pelvis
        r_knee = self.MyoEnv.env.sim.data.joint('knee_angle_r').xanchor[[0,2]].copy() - pelvis
        r_hip = self.MyoEnv.env.sim.data.joint('hip_flexion_r').xanchor[[0,2]].copy() - pelvis

        l_foot = self.MyoEnv.env.sim.data.site('l_foot_touch').xpos[[0,2]].copy() - pelvis
        l_toe = self.MyoEnv.env.sim.data.site('l_toes_touch').xpos[[0,2]].copy() - pelvis

        l_ankle = self.MyoEnv.env.sim.data.joint('ankle_angle_l').xanchor[[0,2]].copy() - pelvis
        l_knee = self.MyoEnv.env.sim.data.joint('knee_angle_l').xanchor[[0,2]].copy() - pelvis
        l_hip = self.MyoEnv.env.sim.data.joint('hip_flexion_l').xanchor[[0,2]].copy() - pelvis


        # R Foot first, then Left foot
        return np.array([r_foot, r_toe, r_ankle, r_knee, r_hip, 
                         l_foot, l_toe, l_ankle, l_knee, l_hip])


    def get_intrinsic_EulerXYZ(self, q):
        w, x, y, z = q

        # Compute sin and cos values
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)

        # Roll (X-axis rotation)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Compute sin and cos values
        sinp = 2 * (w * y - z * x)

        # Pitch (Y-axis rotation)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Compute sin and cos values
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        # Yaw (Z-axis rotation)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)
    
    def _rotate_frame(self, x, y, theta):
        #print(theta)
        x_rot = np.cos(theta)*x - np.sin(theta)*y
        y_rot = np.sin(theta)*x + np.cos(theta)*y
        return x_rot, y_rot

    def convertRange(self, pre_value, old_range, new_range):
        """
        Maps the value to a new range
        old_ramge : nparray with 2 old ranges (min, max)
        new_ramge : nparray with 2 new ranges (min, max)
        """
        pre_Range = (old_range[1] - old_range[0])
        new_Range = (new_range[1] - new_range[0])
        new_Value = (((pre_value - old_range[0]) * new_Range) / pre_Range) + new_range[0]

        return np.array(new_Value)

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]
    
    def _get_com(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def collect_reflex_rollouts(self):
        """
        This function collects a full gait cycle with the currently best Reflex controller parameter and stores it internally
        """
        # 20 Jan 2024 - Only for 2D currently
        # 20 Jan 2024 - Testing with return values
        # Capture Sequence:
        #   - Current state
        #   - ReflexCtrl state before sensor data update (Running the step() function normally also updates the reflex ctrl)
        #   - Acts to get to next state

        state_list = []
        reflex_list = []
        act_array = np.zeros( (2000, 22) )
        grf_array = np.zeros( (2000, 2) )

        for rollout_step in range(2000): # Collect 20 sec before extracting

            state_list.append(self.MyoEnv.unwrapped.get_env_state())
            reflex_list.append(copy.deepcopy(self.ReflexCtrl))

            temp_right = (self.MyoEnv.env.sim.data.sensor('r_foot').data[0].copy() + self.MyoEnv.env.sim.data.sensor('r_toes').data[0].copy())
            temp_left = (self.MyoEnv.env.sim.data.sensor('l_foot').data[0].copy() + self.MyoEnv.env.sim.data.sensor('l_toes').data[0].copy())
            grf_array[rollout_step, 0] = temp_right / (np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8)
            grf_array[rollout_step, 1] = temp_left / (np.sum(self.MyoEnv.env.sim.model.body_mass)*9.8)

            self.get_reflex_sensData()
            new_act = self.reflex2mujoco(self.update_reflex_ctr(self.SENSOR_DATA))

            self.MyoEnv.step(new_act)
            
            self.update_footstep()

            act_array[rollout_step, :] = new_act.copy()
            #act_array = np.vstack( (act_array, new_act.copy()) )

        return state_list, reflex_list, act_array, grf_array
    
    def set_reflex_env_state(self, state_dict):
        """
        Overriden function of Myosuite env_base
        """
        #time = state_dict['time']
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        act = state_dict['act'] if 'act' in state_dict.keys() else None
        self.MyoEnv.sim.set_state(qpos=qp, qvel=qv, act=act)
        self.MyoEnv.sim_obsd.set_state(qpos=qp, qvel=qv, act=act)
        if self.MyoEnv.sim.model.nmocap>0:
            self.MyoEnv.sim.data.mocap_pos[:] = state_dict['mocap_pos']
            self.MyoEnv.sim.data.mocap_quat[:] = state_dict['mocap_quat']
            self.MyoEnv.sim_obsd.data.mocap_pos[:] = state_dict['mocap_pos']
            self.MyoEnv.sim_obsd.data.mocap_quat[:] = state_dict['mocap_quat']
        if self.MyoEnv.sim.model.nsite>0:
            self.MyoEnv.sim.model.site_pos[:] = state_dict['site_pos']
            self.MyoEnv.sim.model.site_quat[:] = state_dict['site_quat']
            self.MyoEnv.sim_obsd.model.site_pos[:] = state_dict['site_pos']
            self.MyoEnv.sim_obsd.model.site_quat[:] = state_dict['site_quat']
        self.MyoEnv.sim.model.body_pos[:] = state_dict['body_pos']
        self.MyoEnv.sim.model.body_quat[:] = state_dict['body_quat']
        self.MyoEnv.sim.forward()
        self.MyoEnv.sim_obsd.model.body_pos[:] = state_dict['body_pos']
        self.MyoEnv.sim_obsd.model.body_quat[:] = state_dict['body_quat']
        self.MyoEnv.sim_obsd.forward()

    def init_SB_env(self, obs_param, rew_type, stim_mode, tgt_vel_mode, sine_vel_args, target_vel, delta_mode, reward_wt, tgt_field_ver):
        """
        Custom Environment initialization for StableBaseline3
        """
        self.obs_param = obs_param # observation parameterization
        self.rew_type = rew_type # reward function selection
        self.LENGTH0 = 0.85 # Leg length in model
        self.stim_mode = stim_mode

        # Define action and observation space, based on reflex controller
        # Acts:
        # - Reflex gains (For each leg, needs to control each leg independently) (59 ea, total 118 params(actions))
        # 0:59 - Right (End index exclusive when slicing)
        # 59:118 - Left (End index exclusive when slicing)
        # 118 total (3D), 94 (2D)
        if self.stim_mode == 'reflex':
            # print("Acting on reflex gains")
            if self.delta_control_mode == 'asym':
                if self.mode == '3D':
                    self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(118,), dtype=np.float32)
                elif self.mode == '2D':
                    self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(94,), dtype=np.float32) # 48 38 94 (full list)
            elif self.delta_control_mode == 'sym':
                if self.mode == '3D':
                    self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(59,), dtype=np.float32)
                elif self.mode == '2D':
                    self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(47,), dtype=np.float32) #24 19 47 (full list)
        elif self.stim_mode == 'muscle':
            # print("Acting on muscle stimulations")
            # Only delta action on muscles
            self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(22,), dtype=np.float32)

        # Obs:
        # - Body (9) - Pelvis height, 3x kinematics, 3x kine vel, 2x pelvis vel
        # - Joint angles (2D: 6 3D: 8) (2x hip_add, 2x hip_flex, 2x knee, 2x ank)
        # - Joint velocity (2D: 6 3D: 8) (2x hip_add, 2x hip_flex, 2x knee, 2x ank)
        # - GRF (2)
        # - vtgt field: 242 (2x11x11) *Not now*
        # 293 in total (3D)
        if self.mode == '3D':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(415,), dtype=np.float32)
        elif self.mode == '2D':

            # action_space_delta = 0
            # if self.obs_param is not None:
            #     # Parameterization of obs_dict
            #     if 'spinal_phase' in self.obs_param:
            #         action_space_delta += 22
            #     if 'mus_f' in self.obs_param:
            #         action_space_delta += 22
            #     if 'mus_l' in self.obs_param:
            #         action_space_delta += 22
            #     if 'mus_v' in self.obs_param:
            #         action_space_delta += 22
            #     if 'mus_act' in self.obs_param:
            #         action_space_delta += 22
            # print(f"Total delta: {action_space_delta}")
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(107,), dtype=np.float32) # 46 Wang et al
        """
        Target velocity for testing
        """
        self.tgt_vel_mode = tgt_vel_mode
        # print(f"Environment mode: {tgt_vel_mode}")

        self.sine_vel_args = sine_vel_args

        # Defaults for sine velocities
        self.sine_min = 0.8
        self.sine_max = 1.8
        self.sine_period = 2000 # 20 sec at 0.01 sec dt
        self.phase_shift = 0

        # Randomizer
        # - 0 for eval mode
        # - 1 to 4 for randomized tgt velocities
        self.target_vel_randomizer = 0
        self.target_vel_type = 'constant' #default

        # Default values
        self.target_x_vel = 0
        self.target_y_vel = 0
        self.eval_x_vel = 0
        self.eval_y_vel = 0

        # target velocity mode only expects 'eval' and 'train'
        # For Eval mode, target velocities can be modified externally
        if self.tgt_vel_mode == 'eval':
            self.target_vel_randomizer = 0
            # Init vel
            if target_vel[0] == -1:
                self.eval_x_vel = 1.2
                self.eval_y_vel = 0
                # print("No target velocity. Defaulting to 1.2 for evaluation")
            else:
                self.eval_x_vel = target_vel[0]
                self.eval_y_vel = target_vel[1]
                # print(f"Fixed target velocity: [{self.eval_x_vel}, {self.eval_y_vel}]")
        elif self.tgt_vel_mode == 'train_constant':
                self.target_x_vel = target_vel[0]
                self.target_y_vel = target_vel[1]

        self.tgt_field_ver = tgt_field_ver

        # self.vtgt = VTgtField(visualize=False, version=self.tgt_field_ver, dt=self.dt, seed=self.seed)
        # self.obs_vtgt_space = self.vtgt.vtgt_space
    
        """
        30 hz delay mode
        """
        self.delta_mode = delta_mode
        # print(f"Environment mode: {delta_mode}")

        """
        Storing previous action
        """
        self.prev_action = np.zeros(self.action_space.shape)

        """
        Initializing reward function weights dict
        """
        if reward_wt is not None:
            self.reward_wt = reward_wt
            # print("Using weights from arguments")
        else:
            self.reward_wt = self.DEFAULT_RWD_KEYS_AND_WEIGHTS
            # print("Using default reward items and weights")

        """
        Initial State randomization from saved gait cycles
        """
        # self.init_state_list
        # self.init_reflex_list
        self.init_rnd_state_len = self.init_state_list.shape[0]

        """
        Debug reward func
        """
        self.debug_reward_dict = []
        self.debug_actions = np.zeros(94,)
        self.step_vel = 0
        self.avg_vel = 0

    # ----- Internal plotting functions -----

    def get_plot_data(self, mus_stim):
        
        plot_data = {}
        plot_data['mus_stim'] = mus_stim
        plot_data['mus_act'] = self.MyoEnv.sim.data.act.copy()

        if self.footstep['new']:
            plot_data['new_step'] = 1
        else:
            plot_data['new_step'] = 0

        plot_data['body'] = {}
        plot_data['r_leg'] = {}
        plot_data['l_leg'] = {}

        body_xquat = self.MyoEnv.sim.data.body('pelvis').xquat.copy()

        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        pelvis_euler_vel = self.MyoEnv.sim.data.object_velocity('pelvis', 'body', local_frame=True).copy()

        plot_data['body']['theta'] = pelvis_euler[1] # Forward tilt (+) after conversion
        plot_data['body']['dtheta'] = -1*pelvis_euler_vel[1][2] # velocity about z-axis (z-axis points to the right of the model), forward (-)
        plot_data['body']['theta_f'] = pelvis_euler[0] - np.deg2rad(90) # Right list (+), Left list (-)
        plot_data['body']['dtheta_f'] = pelvis_euler_vel[1][0] # Right list (+)

        # GRF from foot contact sensor values
        temp_right = (self.MyoEnv.sim.data.sensor('r_foot').data[0].copy() + self.MyoEnv.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.MyoEnv.sim.data.sensor('l_foot').data[0].copy() + self.MyoEnv.sim.data.sensor('l_toes').data[0].copy())

        plot_data['r_leg']['load_ipsi'] = temp_right / (np.sum(self.MyoEnv.sim.model.body_mass)*9.8)
        plot_data['l_leg']['load_ipsi'] = temp_left / (np.sum(self.MyoEnv.sim.model.body_mass)*9.8)

        temp_supraspinal_command = copy.deepcopy(getattr(self.ReflexCtrl, 'supraspinal_command'))
        temp_spinal_control_phase = copy.deepcopy(getattr(self.ReflexCtrl, 'spinal_control_phase'))
        temp_moduleOutputs = copy.deepcopy(getattr(self.ReflexCtrl, 'moduleOutputs'))

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            # GRF data for spinal phrases
            plot_data[s_leg]['contact_ipsi'] = 1 if plot_data[s_leg]['load_ipsi'] > 0.1 else 0
            plot_data[s_leg]['contact_contra'] = 1 if plot_data[s_legc]['load_ipsi'] > 0.1 else 0
            plot_data[s_leg]['load_contra'] = plot_data[s_legc]['load_ipsi']

            # Joint angles
            plot_data[s_leg]['joint'] = {}
            plot_data[s_leg]['joint']['hip'] = (np.pi - self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            plot_data[s_leg]['joint']['knee'] = (np.pi + self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            plot_data[s_leg]['joint']['ankle'] = (0.5*np.pi - self.MyoEnv.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0].copy())

            plot_data[s_leg]['d_joint'] = {}
            plot_data[s_leg]['d_joint']['hip'] = self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel[0].copy() 
            plot_data[s_leg]['d_joint']['knee'] = self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel[0].copy()
            plot_data[s_leg]['d_joint']['ankle'] = self.MyoEnv.sim.data.joint(f"ankle_angle_{s_leg[0]}").qvel[0].copy()

            plot_data[s_leg]['joint_torque'] = {}
            plot_data[s_leg]['joint_torque']['hip'] = self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qfrc_constraint[0].copy() + self.MyoEnv.sim.data.joint(f"hip_flexion_{s_leg[0]}").qfrc_smooth[0].copy()
            plot_data[s_leg]['joint_torque']['knee'] = self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_constraint[0].copy() + self.MyoEnv.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_smooth[0].copy()
            plot_data[s_leg]['joint_torque']['ankle'] = self.MyoEnv.sim.data.joint(f"ankle_angle_{s_leg[0]}").qfrc_constraint[0].copy() + self.MyoEnv.sim.data.joint(f"ankle_angle_{s_leg[0]}").qfrc_smooth[0].copy()

            # Check sign - BODY FRAME ALPHA
            plot_data[s_leg]['alpha'] = plot_data[s_leg]['joint']['hip'] - 0.5*plot_data[s_leg]['joint']['knee']
            plot_data[s_leg]['dalpha'] = -1*plot_data[s_leg]['d_joint']['hip'] - 0.5*plot_data[s_leg]['d_joint']['knee'] # Hip Flexion Vel (-) Only for dalpha calculations
            
            if self.mode == '3D':
                plot_data[s_leg]['alpha_f'] = (self.MyoEnv.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy()) + 0.5*np.pi

            temp_mus_force = self.MyoEnv.sim.data.actuator_force.copy()
            temp_mus_len = self.MyoEnv.sim.data.actuator_length.copy()
            temp_mus_vel = self.MyoEnv.sim.data.actuator_velocity.copy()

            temp_mus = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']

            for MUS in temp_mus:
                plot_data[s_leg][MUS] = {}
                plot_data[s_leg][MUS]['f'] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] / (self.muscle_Fmax[s_leg][MUS]) )
                plot_data[s_leg][MUS]['l'] = ( temp_mus_len[self.muscles_dict[s_leg][MUS]] - self.muscle_LT[s_leg][MUS] ) / self.muscle_L0[s_leg][MUS]
                plot_data[s_leg][MUS]['v'] = temp_mus_vel[self.muscles_dict[s_leg][MUS]] / self.muscle_L0[s_leg][MUS]

                # Capturing non-normalized forces as well for comparison
                plot_data[s_leg][MUS]['nonNormalized_f'] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] )
                plot_data[s_leg][MUS]['fmax'] = self.muscle_Fmax[s_leg][MUS]
                plot_data[s_leg][MUS]['L0'] = self.muscle_L0[s_leg][MUS]

            plot_data[s_leg]['supraspinal_command'] = temp_supraspinal_command[s_leg]
            plot_data[s_leg]['spinal_control_phase'] = temp_spinal_control_phase[s_leg]
            plot_data[s_leg]['moduleOutputs'] = temp_moduleOutputs[s_leg]
            
        return plot_data

    # def get_reward_dict_old(self, terminated, truncated, current_action):

    #     reward = 0
    #     dt = 0.01
    #     rew_dict = {}

    #     # alive reward
    #     # should be large enough to search for 'success' solutions (alive to the end) first
    #     rew_dict['alive_rew'] = self.d_reward['alive']
    #     # effort ~ muscle fatigue ~ (muscle activation)^2 
    #     # Metabolic cost
    #     ACT2 = 0
    #     temp_leg = ['r_leg', 'l_leg']
    #     temp_act = self._get_muscle_act()
        
    #     for leg in temp_leg:
    #         for MUS in self.muscles_dict[leg].keys():
    #             # np.sum used here because there are multiple muscles in each "bundle"
    #             ACT2 += np.sum(np.square( temp_act[self.muscles_dict[leg][MUS]] ))
        
    #     # Accumulates from timesteps, no reward if no new step, so zero
    #     rew_dict['footstep'] = 0
    #     rew_dict['v_tgt'] = 0
    #     rew_dict['effort'] = 0
    #     rew_dict['terminated'] = 0

    #     # Accumulator
    #     self.d_reward['footstep']['effort'] += ACT2*dt
    #     self.d_reward['footstep']['del_t'] += dt
    #     # reward from velocity (penalize from deviating from v_tgt)
    #     self.d_reward['footstep']['del_v'] += self._get_vel_diff(self.get_pel_xvel())*dt
    #     self.step_vel += self.get_pel_xvel()*dt
    #     self.avg_vel = 0

    #     # footstep reward (when made a new step)
    #     if self.footstep['new']:

    #         if self.d_reward['footstep']['del_t'] == 0:
    #             self.avg_vel = 0
    #         else:
    #             self.avg_vel = (self.step_vel / self.d_reward['footstep']['del_t'])[0]

    #         #print(f"Timestep: {self.time_step} , Avg del_v: {self.d_reward['footstep']['del_v']}, average step vel: {self.avg_vel}, Pel Pos: {self.MyoEnv.unwrapped.sim.data.body('pelvis').xpos.copy()[0]}")
    #         #print(f"Del_t :{self.d_reward['footstep']['del_t']} , Calc avg vel: {self.step_vel / self.d_reward['footstep']['del_t']}, Calc diff : {self.d_reward['footstep']['del_v'] / self.d_reward['footstep']['del_t']}")
    #         # footstep reward: so that solution does not avoid making footsteps
    #         # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
    #         reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

    #         # deviation from target velocity
    #         # the average velocity a step (instead of instantaneous velocity) is used
    #         # as velocity fluctuates within a step in normal human walking
    #         # Scale reward by velocity difference distance to zero
    #         reward_footstep_v = -1*self.d_reward['weight']['v_tgt']*(np.linalg.norm(self.d_reward['footstep']['del_v']))
            
    #         #avg_diff = self.d_reward['footstep']['del_v'] / self.d_reward['footstep']['del_t']
    #         #reward_footstep_v = -1*self.d_reward['weight']['v_tgt']*(np.linalg.norm(avg_diff))
    #         #reward_footstep_v = self.d_reward['weight']['v_tgt']*( np.sum( np.exp(-np.square(avg_diff))*0.1 ) )

    #         # panalize effort
    #         reward_footstep_e = -1*self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

    #         self.d_reward['footstep']['del_t'] = 0
    #         self.d_reward['footstep']['del_v'] = 0
    #         self.d_reward['footstep']['effort'] = 0

    #         #reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e
    #         rew_dict['footstep'] = reward_footstep_0
    #         rew_dict['v_tgt'] = reward_footstep_v
    #         rew_dict['effort'] = reward_footstep_e

    #         #self.step_vel = 0
    #         #print(f"Step made: {self.time_step}, step rew: {rew_dict['footstep']}")
    #         #print(f"In step: Terminated: {terminated}, trunc: {truncated}")
    #     # success bonus
    #     if terminated and (truncated == False): #and self.failure_mode is 'success':
    #         # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
    #         #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
    #         #reward += reward_footstep_0 + 100
    #         #reward += self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t'] + 10
    #         #rew_dict['terminated'] = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t'] + 10
    #         rew_dict['footstep'] += self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
    #         #print(f"timestep: {self.time_step},  Terminated_footstep: {rew_dict['footstep']}")
    #         #print(f"In terminated: Terminated: {terminated}, trunc: {truncated}")
    #         #print(f"Terminated: {rew_dict['terminated']}")

    #     # Action delta penalty
    #     #reward += np.mean(np.exp(-1000 * current_action**2)-0.2)
    #     # Both are calculated, but only 1 penalty is applied during reward calculation
    #     rew_dict['action_penalty_zero'] = -1*np.mean( np.square(current_action) ) #np.mean(np.exp(-1000 * current_action**2)-0.1)
    #     rew_dict['action_penalty'] = -1*np.mean( np.square(self.prev_action.copy() - current_action) )
        
    #     #self.debug_reward_dict.append(copy.deepcopy(rew_dict))
    #     self.debug_reward_dict = copy.deepcopy(rew_dict)
    #     self.debug_actions = current_action.copy()

    #     return rew_dict

    # def reset_control_params(self):
    #     # Setting control params
    #     if self.mode == '2D':
    #         #flag_ctrl_mode = '2D' # use 2D
    #         #param_num = 47 + 9 # 9 more for pose optimization
    #         reflex_params = self.CONTROL_PARAM[0:47]
    #         pose_params = self.CONTROL_PARAM[47:len(self.CONTROL_PARAM)]
    #         self.update_init_pose_param_cmaes(self.CONTROL_PARAM[47:len(self.CONTROL_PARAM)])
    #     elif self.mode == '3D':
    #         #flag_ctrl_mode = '3D'
    #         #param_num = 59 + 13 # 13 more for pose optimization
    #         reflex_params = self.CONTROL_PARAM[0:59]
    #         pose_params = self.CONTROL_PARAM[59:len(self.CONTROL_PARAM)]
    #         self.update_init_pose_param_cmaes(self.CONTROL_PARAM[59:len(self.CONTROL_PARAM)])

    #     # if self.mode == '2D':
    #     #     self.update_init_pose_param_cmaes(self.CONTROL_PARAM[47:len(self.CONTROL_PARAM)])
    #     # elif self.mode == '3D':
    #     #     self.update_init_pose_param_cmaes(self.CONTROL_PARAM[59:len(self.CONTROL_PARAM)])

    #     self.ReflexCtrl.reset(reflex_params)
    #     # self.adjust_initial_pose_cmaes()
    #     self.adjust_initial_pose_cmaes()
    #     self.adjust_model_height()

    # def update_init_pose_param_cmaes(self,jnt_params):
    #     # Pelvis tilt, height (pelvis_ty)
    #     # hip, knee, ankle
    #     # forward velocity

    #     if self.mode =='2D' and len(jnt_params) != 9:
    #         raise Exception('2D mode: Wrong number of params')
        
    #     if self.mode =='3D' and len(jnt_params) != 13:
    #         raise Exception('3D mode: Wrong number of params')

    #     # Adjusted such that joint angles add up to the initial defined pose
    #     # Angles are in the Mujoco convention
        
    #     self.JNT_OPTIM['joint_angles'] = {}
    #     self.JNT_OPTIM['joint_angles']['pelvis_ty'] = jnt_params[self.pose_map['pelvis_ty']] *0.01 + 0.868 + self.height_offset # *0.1 + 0.778
    #     self.JNT_OPTIM['joint_angles']['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180) # *2*np.pi/180 + (-17*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-5*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
    #     self.JNT_OPTIM['joint_angles']['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-15*np.pi/180)

    #     if self.mode =='3D':
    #         self.JNT_OPTIM['joint_angles']['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         self.JNT_OPTIM['joint_angles']['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
    #         self.JNT_OPTIM['joint_angles']['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         self.JNT_OPTIM['joint_angles']['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

    #     self.JNT_OPTIM['model_vel'] = {}
    #     self.JNT_OPTIM['model_vel']['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4 #*0.2 + 1.3

    # def adjust_initial_pose_cmaes(self):

    #     # Values in radians
    #     for joint_name in self.JNT_OPTIM['joint_angles'].keys():
    #         self.MyoEnv.sim.data.joint(joint_name).qpos[0] = self.JNT_OPTIM['joint_angles'][joint_name]
        
    #     for vel in self.JNT_OPTIM['model_vel'].keys():
    #         tmp_var = vel.split('_')
    #         self.MyoEnv.sim.data.joint(f"{tmp_var[1]}_{tmp_var[2]}").qvel[0] = self.JNT_OPTIM['model_vel'][vel]
    #     # Run forward() after modifying and joint angles or velocities
    #     self.MyoEnv.forward()

    # def _adjust_initial_pose_cmaes(self, jnt_params):
    #     # Pelvis tilt, height (pelvis_ty)
    #     # hip, knee, ankle
    #     # forward velocity

    #     if self.mode =='2D' and len(jnt_params) != 9:
    #         raise Exception('2D mode: Wrong number of params')
        
    #     if self.mode =='3D' and len(jnt_params) != 13:
    #         raise Exception('3D mode: Wrong number of params')

    #     # Adjusted such that joint angles add up to the initial defined pose
    #     # Angles are in the Mujoco convention

    #     jnt_optim = {}
    #     jnt_optim['joint_angles'] = {}
    #     jnt_optim['joint_angles']['pelvis_ty'] = jnt_params[self.pose_map['pelvis_ty']] *0.01 + 0.868
    #     jnt_optim['joint_angles']['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180)
    #     jnt_optim['joint_angles']['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
    #     jnt_optim['joint_angles']['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
    #     jnt_optim['joint_angles']['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
    #     jnt_optim['joint_angles']['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-5*np.pi/180)
    #     jnt_optim['joint_angles']['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
    #     jnt_optim['joint_angles']['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-15*np.pi/180)

    #     if self.mode =='3D':
    #         jnt_optim['joint_angles']['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         jnt_optim['joint_angles']['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
    #         jnt_optim['joint_angles']['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         jnt_optim['joint_angles']['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

    #     jnt_optim['model_vel'] = {}
    #     jnt_optim['model_vel']['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4
        
    #     # Values in radians
    #     for joint_name in jnt_optim['joint_angles'].keys():
    #         self.MyoEnv.env.sim.data.joint(joint_name).qpos[0] = jnt_optim['joint_angles'][joint_name]
        
    #     for vel in jnt_optim['model_vel'].keys():
    #         tmp_var = vel.split('_')
    #         self.MyoEnv.env.sim.data.joint(f"{tmp_var[1]}_{tmp_var[2]}").qvel[0] = jnt_optim['model_vel'][vel]
    #     # Run forward() after modifying and joint angles or velocities
    #     self.MyoEnv.env.forward()