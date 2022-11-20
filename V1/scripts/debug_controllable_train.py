import os
import threading

"""
Debug the 2nd function: How to train controllable agents
We first start with fixed teammates
"""

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-8x8-4p-6f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "debug"
    test_schedule = "debug"
    name = "debug_controllable_train_1"
    seeds = [0, 1, 2, 3]
    cuda_device = [0, 0, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-8x8-4p-4f-v1_2022-11-14_16:15:40.710435_test_crp_0/0"
    #checkpoint_path = ""
    #once_gen_num = 4
    #teammate_t_max = 10000
    iterations = 2 # do not need train controllable agents


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
                --test_function2=True\
                --recorder_load_path={recorder_load_path}\
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
        
