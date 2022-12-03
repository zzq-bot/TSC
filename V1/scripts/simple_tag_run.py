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
    name = "simple_tag_run_iter_10_ogn_5_mlp_ns_ttmax_550000_tmax_1050000_zdim_32_64_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [4, 4, 4]
    recorder_load_path = ""
    test_function2 = False
    #recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-11-27_21_12_27.191572_test_crp_6/0"
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = .5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"

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
    name = "simple_tag_run_iter_10_ogn_5_mlp_ns_ttmax_550000_tmax_1050000_zdim_4_8_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [5, 5, 5]
    recorder_load_path = ""
    test_function2 = False
    #recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-11-27_21_12_27.191572_test_crp_6/0"
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8
    xi = .5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"

if False:
    #TODO
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "simple_tag_vanilla_qmix_iter_10_ogn_5_ttmax_550000_tmax_1050000_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 1, 2]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 0
    team_z_dim = 0
    xi = .5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"

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
    name = "simple_tag_ablation_wo_cl_iter_10_ogn_5_ttmax_550000_tmax_1050000_zdim_32_64_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [3, 4, 5]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = .5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/SimpleTag-1good-3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-11-29_19_27_31.386069_test_crp_0/0"



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
            --proxy_z_dim={proxy_z_dim}\
            --team_z_dim={team_z_dim}\
            --xi={xi}\
            --teammate_t_max={teammate_t_max}\
            --t_max={t_max}\
            --once_gen_num={once_gen_num}\
            --pretrain_enc_path={pretrain_enc_path}\
            --pretrain_teammate_path={pretrain_teammate_path}\
            --test_recorder_load_path={test_recorder_load_path}\
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