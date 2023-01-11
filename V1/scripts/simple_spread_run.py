import os, threading


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_spread_run_iter_40_ogn_1_ttmax_450000_tmax_312500_gru_zdim_4_6_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = ""
    iterations = 10*4
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 450000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_spread_run_iter_40_ogn_1_ttmax_450000_tmax_312500_gru_zdim_16_20_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = ""
    iterations = 10*4
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 16
    team_z_dim = 20
    xi = 0.5
    teammate_t_max = 450000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_spread_run_iter_40_ogn_1_ttmax_250000_tmax_312500_gru_zdim_4_6_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [7, 7, 7]
    recorder_load_path = ""
    iterations = 10*4
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 250000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_spread_run_wo_cl_iter_40_ogn_1_ttmax_450000_tmax_312500_gru_zdim_4_6_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [2, 2, 2]
    recorder_load_path = ""
    iterations = 10*4
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = False
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 450000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    name = "simple_spread_run_vanilla_iter_40_ogn_1_ttmax_450000_tmax_312500_xi_0.5"
    seeds = [0, 2, 3, 4, 5]
    cuda_device = [1, 1, 2, 2, 3]
    recorder_load_path = ""
    iterations = 10*4
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 450000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "mpe:SimpleSpread-4-v0"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    name = "nnnnn"
    seeds = [0, 1, 2]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed0_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.338319_test_crp_0/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 6
    xi = 0.5
    teammate_t_max = 450000
    t_max = 312500 #1250000
    once_gen_num = 1
    pretrain_teammate_path = "pretrain_checkpoint/simple_spread_4agent/pretrain_teammate_path"
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/simple_spread_pretrain_seed2_mpe_SimpleSpread-4-v0_2022-12-26_12_00_28.337750_test_crp_0/0"
    teammate_diversity_reg = False


program_info = __file__

def one_train(remark, cuda_idx, seed):
    cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
            python src/main.py --config={config}\
            --env-config={env_config}\
            with env_args.time_limit={time_limit}\
            env_args.key={key}\
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