# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    if "state_noise_prob" not in cfg["env"]:
        cfg["env"]["state_noise_prob"] = 0.0
    if "state_switch_prob" not in cfg["env"]:
        cfg["env"]["state_switch_prob"] = 0.0

    cfg["env"]["reweight"] = args.reweight
    cfg["env"]["disable_time_reweight"] = args.disable_time_reweight
    cfg["env"]["reweight_alpha"] = args.reweight_alpha
    cfg["env"]["state_search_to_align_reward"] = args.state_search_to_align_reward
    cfg["env"]["eval_randskill"] = args.eval_randskill
    cfg["env"]["enable_buffernode"] = args.enable_buffernode
    cfg["env"]["enable_rand_target_obs"] = args.enable_rand_target_obs
    cfg["env"]["enable_future_target_obs"] = args.enable_future_target_obs
    cfg["env"]["enable_text_obs"] = args.enable_text_obs
    cfg["env"]["enable_nearest_vector"] = args.enable_nearest_vector
    cfg["env"]["enable_obj_keypoints"] = args.enable_obj_keypoints
    cfg["env"]["enable_ig_scale"] = args.enable_ig_scale
    cfg["env"]["enable_ig_plus_reward"] = args.enable_ig_plus_reward
    cfg["env"]["enable_wrist_local_obs"] = args.enable_wrist_local_obs
    cfg["env"]["show_current_traj"] = args.show_current_traj
    cfg["env"]["use_delta_action"] = args.use_delta_action
    cfg["env"]["use_res_action"] = args.use_res_action
    cfg["env"]["obj_rand_scale"] = args.obj_rand_scale
    cfg["env"]["applyDisturbance"] = args.obj_rand_force
    cfg["env"]["enableDisgravity"] = args.enable_disgravity
    cfg["env"]["enableDofObs"] = args.enable_dof_obs
    cfg["env"]["enableEarlyTermination"] = args.enable_early_termination
    cfg["env"]["build_blender_motion"] = args.build_blender_motion
    cfg["env"]["blender_motion_length"] = args.blender_motion_length
    cfg["env"]["blender_motion_name"] = args.blender_motion_name

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True

    if args.checkpoint != "Base":
        cfg_train["params"]["load_path"] = args.checkpoint
        
    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations
    
    if cfg_train["params"]["model"]["name"] == "skillmimic_denseobj":
        cfg["env"]["enable_dense_obj"] = True
    else:
        cfg["env"]["enable_dense_obj"] = False

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = args

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--projtype", "type": str, "default": "None",
            "help": "Can be None, Auto, Mouse"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", #ZT9 "cpu"
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_env", "type": str, "default": "Base", "help": "Environment configuration file (.yaml)"},
        {"name": "--cfg_train", "type": str, "default": "Base", "help": "Training configuration file (.yaml)"},
        {"name": "--motion_file", "type": str,
            "default": "", "help": "Specify reference motion file"},
        {"name": "--play_dataset", "action": "store_true", "default": False,
            "help": "Display the dataset"},
        {"name": "--postproc_unihotdata", "action": "store_true", "default": False,
            "help": "Postprocess the raw dataset"},
        {"name": "--init_vel", "action": "store_true", "default": False,
            "help": "Init the object velocity at the first frame"},
        {"name": "--save_images", "action": "store_true", "default": False,
            "help": "save images for viewer"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--frames_scale", "type": float, "default": 0.,
            "help": "Set the fps scale for the reference HOI dataset"},
        {"name": "--ball_size", "type": float, "default": 0.,
            "help": "Set the ball size scale"},
        {"name": "--cg1", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 1"},
        {"name": "--cg2", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 2"},
        {"name": "--ig", "type": float, "default": -1.,
            "help": "Set the interaction graph reward weight"},
        {"name": "--op", "type": float, "default": -1.,
            "help": "Set the object position reward weight"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--output_path", "type": str, "default": "output/", "help": "Specify output directory"},
        {"name": "--llc_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."},
        {"name": "--resume_from", "type": str, "default": "", 
            "help": "Specify the checkpoint to continue training"},
        {"name": "--state_init", "type": str, "default": "Random", 
            "help": "Specify a specific initialization frame and disable random initialization. Or Random Reference State Init"},
        {"name": "--reweight", "action": "store_true", "default": False,
            "help": "Specify whether to reweight the motion_id and motion_time according to the reward."},
        {"name": "--reweight_alpha", "type": float, "default": 0., 
            "help": "Specify the reweight extend."},
        {"name": "--disable_time_reweight", "action": "store_true", "default": False,
            "help": "Specify whether to disable the time reweight."},
        {"name": "--state_noise_prob", "type": float, "default": 0, 
            "help": "Specify the random init probability"},
        {"name": "--state_switch_prob", "type": float, "default": 0, 
            "help": "Specify the random init probability"},
        {"name": "--state_search_to_align_reward", "action": "store_true", "default": False,
            "help": "Specify whether to align frame-wise rewards by searching for similar states during randomized initialization."},
        {"name": "--eval_randskill", "action": "store_true", "default": False,
            "help": "Specify whether to evaluate random skill while playing the dataset"},
        {"name": "--enable_buffernode", "action": "store_true", "default": False,
            "help": "Enable or disable the buffernode functionality."},
        {"name": "--enable_rand_target_obs", "action": "store_true", "default": False,
            "help": "Enable or disable the random target observation functionality."},
        {"name": "--enable_future_target_obs", "action": "store_true", "default": False,
            "help": "Enable or disable the future target observation functionality."},
        {"name": "--enable_text_obs", "action": "store_true", "default": False,
            "help": "Enable or disable the skill label observation functionality."},
        {"name": "--enable_nearest_vector", "action": "store_true", "default": False,
            "help": "Enable or disable the nearest vector observation functionality."},
        {"name": "--enable_obj_keypoints", "action": "store_true", "default": False,
            "help": "Enable or disable the object keypoints observation functionality."},
        {"name": "--enable_ig_scale", "action": "store_true", "default": False,
            "help": "Enable or disable the scaling for the reward computation of interaction graph."},
        {"name": "--enable_ig_plus_reward", "action": "store_true", "default": False,
            "help": "Enable or disable the object to keypoints relative position reward."},
        {"name": "--enable_wrist_local_obs", "action": "store_true", "default": False,
            "help": "Enable or disable the object to keypoints relative position reward."},
        {"name": "--show_current_traj", "action": "store_true", "default": False,
            "help": "Set it true to show current trajectory, otherwise show the next frame trajectory."},
        {"name": "--use_delta_action", "action": "store_true", "default": False,
            "help": "Set it true to use delta action, otherwise use absolute action."},
        {"name": "--use_res_action", "action": "store_true", "default": False,
            "help": "Set it true to use residual action, otherwise use absolute action."},
        {"name": "--obj_rand_scale", "action": "store_true", "default": False,
            "help": "Set it true to use object random scale, otherwise use fixed scale."},
        {"name": "--obj_rand_force", "action": "store_true", "default": False,
            "help": "Set it true to use object random force."},
        {"name": "--enable_disgravity", "action": "store_true", "default": False,
            "help": "Set it true to set the disgravity for object."},\
        {"name": "--enable_dof_obs", "action": "store_true", "default": False,
            "help": "Set it true to set the disgravity for object."},
        {"name": "--enable_early_termination", "action": "store_true", "default": False,},
        {"name": "--objnames", "nargs": "*", "type": str, "default": None,
            "help": "Object names to use in the environment. Can accept multiple values, a single value, or be omitted"},
        {"name": "--build_blender_motion", "action": "store_true", "default": False,
            "help": "Enable or disable the building of blender motion."},
        {"name": "--blender_motion_length", "type": int, "default": 0,
            "help": "Set the length of the blender motion."},
        {"name": "--blender_motion_name", "type": str, "default": "",
            "help": "Set the filename of the blender motion."}
    ]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
