import os
import threading

"""
Here we test env of simple tag
Allowance for predator and prey:
    (1, 3),
    (1, 4),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5)
    f"SimpleTag-{num_good}good-{num_adv}adv-v0"
"""

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "test_env_generate"
    seeds = [0]
    cuda_device = [6]
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
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "train"
    name = "test_env_controllable"
    seeds = [0]
    cuda_device = [6]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-11-27_21_12_27.191572_test_crp_6/0"
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
            env_args.pretrained_wrapper={pretrained_wrapper}\
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