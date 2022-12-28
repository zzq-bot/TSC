import os
import threading

"""
Here we pretrain teammates checkpoint
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
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "simple_spread_pretrain"
    #seeds = [0, 1, 2, 3, 4]
    #cuda_device = [0, 0, 0, 0]
    seeds = [4, 5]
    cuda_device = [7, 8]
    recorder_load_path = ""
    test_function2 = False  
    iterations = 1
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    once_gen_num = 6
    xi = 1.5
    teammate_t_max = 1450000


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "simple_spread_pretrain_2"
    seeds = [6, 7, 8, 9, 10, 11]
    cuda_device = [4, 4, 4, 5, 5, 5]
    recorder_load_path = ""
    test_function2 = False  
    iterations = 1
    z_gen_hyper = False
    agent = "mlp_ns"
    once_gen_num = 5
    xi = 1.2
    teammate_t_max = 2550000

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
            --xi={xi}\
            --teammate_agent={teammate_agent}\
            --teammate_t_max={teammate_t_max}&\
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