import os
import threading

"""
Debug
"""
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    run_type = "debug_1"
    key = "Foraging-6x6-4p-3f-coop-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_performance_drop"
    seeds = [0]
    cuda_device = [0]
    test_function2 = False
    z_gen_hyper = False
    use_contrastive_loss = False
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"
    debug_model_path = "debug/performance_drop/lbf6643_coop"
    waiting_lower = 3
    waiting_upper = 6
    render_save_path = "render_save_path/lbf"
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    run_type = "evaluate_pretrain"
    key = "Foraging-6x6-4p-3f-coop-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "debug_performance_drop"
    seeds = [0]
    cuda_device = [0]
    test_function2 = False
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    use_contrastive_loss = False
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed3_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.581723_teammate_pretrain_7/0"
    debug_model_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    waiting_lower = 5
    waiting_upper = 8
    render_save_path = "render_save_path/lbf_6643_coop"

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
    waiting_lower = 3
    waiting_upper = 6
    render_save_path = "render_save_path/mpe_simple_tag"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 30
    run_type = "debug_1"
    key = "Foraging-6x6-4p-3f-v1"
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
    recorder_load_path = "recorder_checkpoint/lbf_6643_teammate_pretrain_seed1_Foraging-6x6-4p-3f-v1_2022-12-13_16_28_07.043569_teammate_pretrain_0/0"
    debug_model_path = "debug/performance_drop/lbf6643"
    waiting_lower = 2
    waiting_upper = 4
    render_save_path = "render_save_path/lbf6643"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    run_type = "evaluate_pretrain"
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
    debug_model_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_teammate_path"
    waiting_lower = 5
    waiting_upper = 8
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
                --teammate_agent={teammate_agent}\
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