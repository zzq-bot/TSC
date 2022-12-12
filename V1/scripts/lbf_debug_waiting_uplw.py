import os
import threading

"""
Debug, what kind of waiting up/lw can lower the performance of vanilla QMIX
"""


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "lbf_debug_waiting_l5u7"
    seeds = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    t_max = 5050000
    waiting_lower = 5
    waiting_upper = 7

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "lbf_debug_waiting_l4u6"
    seeds = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    t_max = 5050000
    waiting_lower = 4
    waiting_upper = 6

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "lbf_debug_waiting_l3u6"
    seeds = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    t_max = 5050000
    waiting_lower = 3
    waiting_upper = 6

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
                --t_max={t_max}\
                --waiting_lower={waiting_lower}\
                --waiting_upper={waiting_upper}\
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