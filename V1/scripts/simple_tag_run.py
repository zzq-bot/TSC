import os
import threading


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    #name = "simple_tag_run_iter_10_ogn_4_ttmax_750000_tmax_1250000_lstm_zdim_32_64_xi_2.5"
    name = "simple_tag_run_iter_10_ogn_4_ttmax_750000_tmax_1250000_lstm_zdim_32_64_xi_1"
    seeds = [0, 1, 2]
    cuda_device = [5, 6, 7]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    #xi = 2.5
    xi = 1
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

#TODO
if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_tag_run_iter_10_ogn_4_ttmax_750000_tmax_1250000_gru_zdim_32_64_xi_2.5"
    seeds = [0, 1, 2]
    cuda_device = [7, 7, 7]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

#TODO
if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_tag_run_wo_cl_iter_10_ogn_4_ttmax_750000_tmax_1250000_lstm_zdim_32_64_xi_2.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

#TODO
if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_tag_run_vanilla_iter_10_ogn_4_ttmax_750000_tmax_1250000_xi_2.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "simple_tag_run_tmax_6250000_lstm_zdim_32_64"
    seeds = [0, 1, 2]
    cuda_device = [4, 4, 4]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.461902_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "simple_tag_run_tmax_6250000_gru_zdim_32_64"
    seeds = [0, 1, 2]
    cuda_device = [5, 5, 5]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.461902_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "simple_tag_run_wo_cl_tmax_6250000_lstm_zdim_32_64"
    seeds = [0, 1, 2]
    cuda_device = [6, 6, 6]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.461902_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "simple_tag_run_vanilla_tmax_6250000"
    seeds = [0, 1, 2]
    cuda_device = [7, 7, 7]
    recorder_load_path = "recorder_checkpoint/test_env_generate_seed0_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.461902_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "debug_simple_tag_run_tmax_6250000_lstm_zdim_32_64"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = "recorder_checkpoint/simple_tag_run_iter_10_ogn_4_ttmax_750000_tmax_1250000_lstm_zdim_32_64_xi_2.5_seed0_mpe_SimpleTag-1good-3adv-v0_2022-12-30_08_52_52.601890_test_crp_5/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = 2.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"

#TODO
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "debug_simple_tag_run_iter_10_ogn_4_ttmax_750000_tmax_1250000_gru_zdim_4_6_xi_0.5"
    seeds = [0, 1]
    cuda_device = [0, 0]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 750000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"
    teammate_diversity_reg = False


if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleTag-1good-3adv-v0"
    pretrained_wrapper = "PretrainedTag"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "debug_simple_tag_run_iter_10_ogn_1_ttmax_750000_tmax_312500_gru_zdim_4_6_xi_0.5"
    seeds = [0, 1]
    cuda_device = [0, 1]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 750000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_tag_3adv/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/test_env_generate_seed2_mpe_SimpleTag-1good-3adv-v0_2022-12-26_19_58_40.376907_test_crp_0/0"
    teammate_diversity_reg = False

    
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
            --test_function2={test_function2}\
            --name={name}\
            --seed={seed}\
            --remark={remark}\
            --recorder_load_path={recorder_load_path}\
            --iterations={iterations}\
            --z_gen_hyper={z_gen_hyper}\
            --agent={agent}\
            --teammate_agent={teammate_agent}\
            --proxy_encoder={proxy_encoder}\
            --team_encoder={team_encoder}\
            --use_contrastive_loss={use_contrastive_loss}\
            --proxy_z_dim={proxy_z_dim}\
            --team_z_dim={team_z_dim}\
            --xi={xi}\
            --teammate_t_max={teammate_t_max}\
            --t_max={t_max}\
            --once_gen_num={once_gen_num}\
            --pretrain_teammate_path={pretrain_teammate_path}\
            --pretrain_enc_path={pretrain_enc_path}\
            --test_recorder_load_path={test_recorder_load_path}\
            --teammate_diversity_reg={teammate_diversity_reg}&\
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