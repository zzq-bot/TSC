import os
import threading

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "simple_tag_debug_waiting_l5u7"
    seeds = [0]
    cuda_device = [1]
    test_function2 = True
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"
    iterations = 2
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
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "simple_tag_debug_waiting_l4u6"
    seeds = [0]
    cuda_device = [1]
    test_function2 = True
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"
    iterations = 2
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
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    name = "simple_tag_debug_waiting_l3u6"
    seeds = [0]
    cuda_device = [1]
    test_function2 = True
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    t_max = 5050000
    waiting_lower = 3
    waiting_upper = 6

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