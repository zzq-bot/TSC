import os
import threading

"""
Baseline Vanilla QMIX for sudden change
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
    name = "debug_baseline_vanilla_qmix"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_ablation_wo_contrastive_loss"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    test_function2 = True
    iterations = 2 # do not need train controllable agents currently
    use_encoder = True
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = False


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