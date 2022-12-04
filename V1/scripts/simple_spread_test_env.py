import os
import threading

"""
Here we test env of simple spread
Allowance for agents:
    3
    4
    5
    6
    f"SimpleSpread-{n}-v0"
"""

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "test_env_generate"
    seeds = [1]
    cuda_device = [1]
    recorder_load_path = ""
    test_function2 = False
    iterations = 1
    z_gen_hyper = False
    agent = "mlp_ns"
    once_gen_num = 4
    teammate_t_max = 10000
    t_max = 100000


if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "train"
    name = "test_spread_controllable"
    seeds = [0]
    cuda_device = [1]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed1_mpe_SimpleSpread-4-v0_2022-12-04_13_43_37.947569_test_crp_1/0"
    test_function2 = True
    iterations = 2
    z_gen_hyper = False
    agent = "mlp_ns"
    once_gen_num = 4
    teammate_t_max = 10000
    t_max = 100000
    
program_info = __file__

def one_train(remark, cuda_idx, seed):
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
            --recorder_load_path={recorder_load_path}\
            --test_function2={test_function2}\
            --iterations={iterations}\
            --z_gen_hyper={z_gen_hyper}\
            --agent={agent}\
            --once_gen_num={once_gen_num}\
            --teammate_t_max={teammate_t_max}\
            --t_max={t_max}&\
            sleep 2s"
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