import os
import threading

"""
Debug
"""
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    run_type = "debug_1"
    key = "Foraging-8x8-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_performance_drop"
    seeds = [0]
    cuda_device = [0]
    test_function2 = False
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"
    debug_model_path = "debug/performance_drop/lbf"
    waiting_lower = 5
    waiting_upper = 8
    render_save_path = "render_save_path/lbf"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    run_type = "debug_1"
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_performance_drop"
    seeds = [0]
    cuda_device = [0]
    test_function2 = False
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"
    debug_model_path = "debug/performance_drop/mpe_stag"
    waiting_lower = 1
    waiting_upper = 2
    render_save_path = "render_save_path/mpe_simple_tag"
    
program_info = __file__

def one_train(remark, cuda_idx, seed):
    if "Foraging" in key:
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- \
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
                --run_type={run_type}\
                --debug_model_path={debug_model_path}\
                --waiting_lower={waiting_lower}\
                --waiting_upper={waiting_upper}\
                --render_save_path={render_save_path}\
                --use_contrastive_loss={use_contrastive_loss} &\
                sleep 2s"
    elif "SimpleTag" in key:
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- \
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
                --run_type={run_type}\
                --debug_model_path={debug_model_path}\
                --waiting_lower={waiting_lower}\
                --waiting_upper={waiting_upper}\
                --render_save_path={render_save_path}\
                --use_contrastive_loss={use_contrastive_loss} &\
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