import datetime
import os
import pprint
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import torch as th
import numpy as np
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from icecream import ic
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from modules.recorders.crp_recorder import CRPRecorder
from modules.traj_encoder.transformer_encoder import TransformerEncoder
from modules.decoder.rnn_decoder import RNNDecoder


def test_run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)
    if "n_control" not in _config:
        _config["n_control"] = None
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}_{_config['remark']}".replace(" ", "_")

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    if args.n_control is None:
        args.n_control = args.n_agents
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme for controllable agents
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "key": {"vshape":(1, ), "dtype": th.long}
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    

    if args.use_cuda:
        learner.cuda()
    
    args.recorder_path = os.path.join(args.recorder_path, args.unique_token)

    crp_recorder = CRPRecorder(args)
    traj_encoder = TransformerEncoder(args)
    traj_decoder = RNNDecoder(args)
    enc_params = list(traj_encoder.parameters()) + list(traj_decoder.parameters())
    enc_optimiser = th.optim.Adam(params=enc_params, lr=args.lr)
    crp_recorder.set_module(traj_encoder, traj_decoder)
    if args.recorder_load_path != "":
        logger.console_logger.info("Load CRP_Recorder from {}".format(args.recorder_load_path))
        crp_recorder.load(load_path=args.recorder_load_path)
    

    if args.test_crp_performance:
        assert args.recorder_load_path != ""

        teammate_buffer = ReplayBuffer(
                    scheme,
                    groups,
                    args.buffer_size,
                    env_info["episode_limit"] + 1,
                    preprocess=preprocess,
                    device="cpu" if args.buffer_cpu_only else args.device,
                )

        teammate_mac = mac_REGISTRY[args.teammate_mac](teammate_buffer.scheme, groups, args)
                
        teammate_runner = r_REGISTRY[args.teammate_runner](args=args, logger=logger)
        teammate_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=teammate_mac)

        teammate_learner = le_REGISTRY[args.teammate_learner](teammate_mac, teammate_buffer.scheme, logger, args)
        teammate_learner.set_enc(traj_encoder, traj_decoder, enc_params, enc_optimiser)
        teammate_learner.set_recorder(crp_recorder)
        if args.use_cuda:
                    teammate_learner.cuda()
    
        for checkpoint in crp_recorder.record_checkpoint_path:
            teammate_learner.load_models(checkpoint)
            logger.console_logger.info(
                "Load teammate model from checkpoint {}".format(checkpoint)
            )
            # Execute test runs
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            for _ in range(n_test_runs):
                teammate_runner.run(test_mode=True)
        teammate_runner.close_env()

    # If test controllable agents
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        # Load controllable model
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load


        
        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    logger.console_logger.info("Finished Testing")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
