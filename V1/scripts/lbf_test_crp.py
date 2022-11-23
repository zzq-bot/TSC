import os
import threading

"""
Just test basic function of CRP Process/Recorder
"""
# python3 src/main.py --config=my_qmix --env-config=gymma with 
# env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-5p-4f-v1" 
# learner='q_learner' use_encoder=False train_schedule="base"
#  test_schedule="fixed_dynamic" name="tr_s-te_fd" seed=$SEED
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-8x8-4p-4f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "base"
    test_schedule = "fixed_dynamic"
    name = "just_test"
    seed = 0
    cuda_device = [0]
    checkpoint_path = ""
    once_gen_num = 4
    teammate_t_max = 10000

program_info = __file__

def one_train(remark, cuda_idx):
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx};\
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
                --checkpoint_path={checkpoint_path}\
                --once_gen_num={once_gen_num}\
                --teammate_t_max={teammate_t_max}"
    else:
        assert 0
    ret = os.system(cmd)
    if ret != 0:
        print("failure !!!!")

if __name__ == "__main__":
    threads = []
    base_remark = "test_crp"
    cuda_indices = [0]
    for cuda_idx in cuda_indices:
        remark = f"{base_remark}_{cuda_idx}"
        th = threading.Thread(target=one_train, args=(remark, cuda_idx))
        th.start()
        threads.append(th)
    
    for th in threads:
        th.join()
        
