import os
import threading

"""
Pretrain and generate some diverse teammates
"""
# python3 src/main.py --config=my_qmix --env-config=gymma with 
# env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-5p-4f-v1" 
# learner='q_learner' use_encoder=False train_schedule="base"
#  test_schedule="fixed_dynamic" name="tr_s-te_fd" seed=$SEED
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "base"
    test_schedule = "fixed_dynamic"
    name = "teammate_pretrain"
    seeds = [0, 1, 2, 3, 4, 5]
    #seeds = [6,7,8,9,10, 11]
    cuda_device = [4, 4, 4, 5, 5, 5]
    #checkpoint_path = ""
    once_gen_num = 10
    teammate_t_max = 2050000
    iterations = 1 # dont need to train controllable agents
    xi=5

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-6f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "base"
    test_schedule = "fixed_dynamic"
    name = "teammate_pretrain"
    seeds = [0,1,2,3,4, 5]
    cuda_device = [0, 0, 0, 1, 1, 1]
    #checkpoint_path = ""
    once_gen_num = 10
    teammate_t_max = 2050000
    iterations = 1 # do not need train controllable agents
    xi=5

program_info = __file__

def one_train(remark, cuda_idx, seed):
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                python3 src/main.py --config={config}\
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
                --once_gen_num={once_gen_num}\
                --teammate_t_max={teammate_t_max}\
                --xi={xi}\
                --iterations={iterations} &\
                sleep 2s"
    else:
        assert 0
    ret = os.system(cmd)
    if ret != 0:
        print("failure !!!!")

if __name__ == "__main__":
    threads = []
    base_remark = "teammate_pretrain"
    
    for seed, cuda_idx in zip(seeds, cuda_device):
        remark = f"{base_remark}_{cuda_idx}"
        th = threading.Thread(target=one_train, args=(remark, cuda_idx, seed))
        th.start()
        threads.append(th)
    
    for th in threads:
        th.join()
        
