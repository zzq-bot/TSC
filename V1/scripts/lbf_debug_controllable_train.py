import os
import threading

"""
Debug the 2nd function: How to train controllable agents
We first start with fixed teammates
"""

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-6f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "debug_controllable_train_1"
    seeds = [0, 1, 2, 3]
    cuda_device = [0, 0, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-6f-v1_2022-11-15_11_31_56.339086_teammate_pretrain_0/0"
    test_function2=True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "debug_controllable_train_2"
    seeds = [0, 1]
    cuda_device = [0, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_controllable_train_3"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_controllable_train_4"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_controllable_train_4"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64

if True:
    # TODO
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_controllable_train_5_with_zdim_4_8"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "check_freeze_debug_controllable_train_3"
    seeds = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64


"""if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_controllable_train_4"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True"""


program_info = __file__

def one_train(remark, cuda_idx, seed):
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                python src/main.py --config={config}\
                --env-config={env_config}\
                with env_args.time_limit={time_limit}\
                env_args.key={key}\
                --learner={learner}\
                --use_encoder={use_encoder}\
                --train_schedule={train_schedule}\
                --test_schedule={test_schedule}\
                --name={name}\
                --seed={seed}\
                --remark={remark}\
                --test_function2={test_function2}\
                --recorder_load_path={recorder_load_path}\
                --z_gen_hyper={z_gen_hyper}\
                --agent={agent}\
                --use_contrastive_loss={use_contrastive_loss}\
                --proxy_z_dim={proxy_z_dim}\
                --team_z_dim={team_z_dim}\
                --iterations={iterations} &\
                sleep 2s"
    else:
        assert 0
    ret = os.system(cmd)
    if ret != 0:
        print("failure !!!!")

if __name__ == "__main__":
    threads = []
    base_remark = "test_crp"
    
    for seed, cuda_idx in zip(seeds, cuda_device):
        remark = f"{base_remark}_{cuda_idx}"
        th = threading.Thread(target=one_train, args=(remark, cuda_idx, seed))
        th.start()
        threads.append(th)
    
    for th in threads:
        th.join()
        
