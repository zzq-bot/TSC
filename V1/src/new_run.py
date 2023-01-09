import datetime
import os
import pprint
import threading
import time
import random
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


def run(_run, _config, _log):
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
    unique_token=  unique_token.replace(':', '_')

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

    # start training
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))


    # We pretrained controllable agents under "my_run" (with heuristic based teammates)

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
        #crp_recorder.save
        crp_recorder.re_save(args)
        recorder_save_path = os.path.join(args.recorder_path, f"{args.start_iter}")
        crp_recorder.save(recorder_save_path)

    test_crp_recorder = None
    if args.test_recorder_load_path != "":
        test_progressive = False
        test_crp_recorder = CRPRecorder(args)
        check_dir = os.path.join(args.test_recorder_load_path, os.listdir(args.test_recorder_load_path)[0])
        if os.path.isdir(check_dir):
            logger.console_logger.info("Load Progressive Test CRP_Recorder from {}".format(args.test_recorder_load_path))
            test_progressive = True
        else:
            logger.console_logger.info("Load Test CRP_Recorder from {}".format(args.test_recorder_load_path))
            test_crp_recorder.load(load_path=args.test_recorder_load_path)
        
    if args.pretrain_enc_path!="":
        logger.console_logger.info("Load EncDec from {}".format(args.pretrain_enc_path))
        traj_encoder.load_state_dict(th.load("{}/encoder.th".format(args.pretrain_enc_path), \
            map_location=lambda storage, loc: storage))
        traj_decoder.load_state_dict(th.load("{}/decoder.th".format(args.pretrain_enc_path), \
            map_location=lambda storage, loc: storage))
    else:
        logger.console_logger.info("Train EncDec from scratch")

    if args.pretrain_teammate_path!="":
        candidate_pretrain_teammate_path_list = os.listdir(args.pretrain_teammate_path)
        random.shuffle(candidate_pretrain_teammate_path_list)
        k = 0

    for i in range(args.start_iter, args.iterations):
        if i % 2 == 0:
            teammate_last_test_T = 0
            if args.test_function2:
                continue
            recorder_save_path = os.path.join(args.recorder_path, f"{i//2}")
            logger.console_logger.info("This iter, CRP_Recorder will be saved in {}".format(recorder_save_path)) 
            for id in range(args.once_gen_num):
                logger.console_logger.info("Now, generate {}/{} teammates".format(id+1, args.once_gen_num)) 
                # Default/Base scheme for controllable agents for teammates generator
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
                teammate_runner.set_id(i, id)
                teammate_learner = le_REGISTRY[args.teammate_learner](teammate_mac, teammate_buffer.scheme, logger, args)
                
                # Load pretrained teammate model (by default)
                if args.pretrain_teammate_path!="":
                    chosen_path = candidate_pretrain_teammate_path_list[k]
                    k = (k + 1) % len(candidate_pretrain_teammate_path_list)
                    logger.console_logger.info("Use Pretrained Teammate Agent Model from {}".format(chosen_path))
                    teammate_learner.load_agent_models(path=os.path.join(args.pretrain_teammate_path, chosen_path))
                else:
                    logger.console_logger.info("Train Teammate Agent Model from scratch")

                teammate_learner.set_enc(traj_encoder, traj_decoder, enc_params, enc_optimiser)
                teammate_learner.set_recorder(crp_recorder)

                if args.use_cuda:
                    teammate_learner.cuda()
            
                episode = 0
                while teammate_runner.t_env <= args.teammate_t_max:
                    episode_batch = teammate_runner.run(test_mode=False)
                    teammate_buffer.insert_episode_batch(episode_batch)

                    if teammate_buffer.can_sample(args.batch_size):
                        episode_sample = teammate_buffer.sample(args.batch_size)

                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)
                        #ic("Learner is going to update")
                        teammate_learner.train(episode_sample, teammate_runner.t_env, episode)
                    # TODO test?
                    # Execute test runs once in a while
                    n_test_runs = max(1, args.test_nepisode // teammate_runner.batch_size)

                    if (teammate_runner.t_env - teammate_last_test_T) / args.test_interval >= 1.0:

                        logger.console_logger.info(
                            "t_env: {} / {}".format(teammate_runner.t_env, args.teammate_t_max)
                        )
                        logger.console_logger.info(
                            "Estimated time left: {}. Time passed: {}".format(
                                time_left(last_time, teammate_last_test_T, teammate_runner.t_env, args.teammate_t_max),
                                time_str(time.time() - start_time),
                            )
                        )
                        last_time = time.time()

                        teammate_last_test_T = teammate_runner.t_env
                        for _ in range(n_test_runs):
                            teammate_runner.run(test_mode=True)
                    episode += args.batch_size_run
                

                # TODO, should we update cluster center as traj_encoder has been updated
                if args.update_cluster_center:
                    crp_recorder.update_cluster_center()
                # determine npc_index
                maximum_npc_num = args.n_agents-args.n_control
                # assert maximum_npc_num == 2
                this_npc_num = np.random.choice(range(1, maximum_npc_num+1), 1).item()
                npc_idx = np.sort(np.random.choice(range(args.n_agents), this_npc_num)) # TODO ?
                print("cur chosen npc_idx is:", npc_idx)
                cluster_buffer = ReplayBuffer(
                    scheme,
                    groups,
                    args.num_traj_cluster,
                    env_info["episode_limit"] + 1,
                    preprocess=preprocess,
                    device="cpu" if args.buffer_cpu_only else args.device,
                )
                # should we use 
                for _ in range(args.num_traj_cluster):
                    #TODO
                    if args.cluster_itself:
                        episode_batch = teammate_runner.run(test_mode=True)
                        cluster_buffer.insert_episode_batch(episode_batch)
                    else:
                        # Use 
                        raise NotImplementedError("TBD. Use traj generated by chosen npcs and controllable agents.") 
                # get X and Y from 
                ic("Here I cluster this teammmate model")
                X, Y, masks = crp_recorder.build_cluster_input(cluster_buffer)
                m, this_teammate_save_path = crp_recorder.cluster(npc_idx, X, Y, masks, args.unique_token)
                ic("this belongs to the cluster:", m)
                teammate_learner.save_models(this_teammate_save_path)
                #del teammate_buffer, teammate_learner, teammate_mac
                crp_recorder.save(recorder_save_path)


        else:
            #assert 0
            logger.console_logger.info("##################################")
            logger.console_logger.info("Start training controllabel agents")
            
            mac.set_schedule_recorder(crp_recorder)
            """if test_crp_recorder is not None:
                mac.set_schedule_recorder(test_crp_recorder, mode='test')
            else:
                mac.set_schedule_recorder(crp_recorder, mode='test')"""
            episode = 0
            while runner.t_env <= args.t_max * (i//2+1):
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    #ic("Learner is going to update")
                    learner.train(episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info(
                        "t_env: {} / {}".format(runner.t_env, args.t_max)
                    )
                    logger.console_logger.info(
                        "Estimated time left: {}. Time passed: {}".format(
                            time_left(last_time, last_test_T, runner.t_env, args.t_max),
                            time_str(time.time() - start_time),
                        )
                    )
                    last_time = time.time()

                    last_test_T = runner.t_env
                    if test_crp_recorder is None:
                        for _ in range(n_test_runs):
                            runner.run(test_mode=True)
                    else:
                        if test_progressive:
                            #TODO, re-load crp_recorder
                            test_recorder_load_path = os.path.join(args.test_recorder_load_path, str(i//2))
                            test_crp_recorder.load(load_path=test_recorder_load_path)
                        mac.set_schedule_recorder(test_crp_recorder, mode='test')
                        for _ in range(n_test_runs):
                            runner.run(test_mode=True)
                        runner.set_prefix("train2test")
                        mac.set_schedule_recorder(crp_recorder, mode='test')
                        for _ in range(n_test_runs):
                            runner.run(test_mode=True)
                        runner.set_prefix("")
                if args.save_model and (
                    runner.t_env - model_save_time >= args.save_model_interval
                    or model_save_time == 0
                ):
                    model_save_time = runner.t_env
                    save_path = os.path.join(
                        args.local_results_path, "models", args.unique_token, str(runner.t_env)
                    )
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env
                
    runner.close_env()
    logger.console_logger.info("Finished Training")


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
