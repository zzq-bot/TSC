import os
import threading

"""
Run the whole process
"""

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_iter_10_ogn_4_ttmax_850000_tmax_1250000_lstm_zdim_32_64_xi_5e-1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
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
    xi =  0.5
    teammate_t_max = 850000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_iter_10_ogn_4_ttmax_850000_tmax_1250000_gru_zdim_32_64_xi_5e-1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
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
    xi =  0.5
    teammate_t_max = 850000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_wo_cl_iter_10_ogn_4_ttmax_850000_tmax_1250000_lstm_zdim_32_64_xi_5e-1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
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
    xi =  0.5
    teammate_t_max = 850000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_vanilla_iter_10_ogn_4_ttmax_850000_tmax_1250000_xi_5e-1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [4, 5, 6]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 850000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_tmax_6250000_lstm_zdim_32_64"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    #cuda_device = [0, 0, 1]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.577800_teammate_pretrain_4/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_tmax_6250000_gru_zdim_32_64"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [1, 2, 2]
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.577800_teammate_pretrain_4/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    name = "lbf_run_wo_cl_tmax_6250000_lstm_zdim_32_64"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    cuda_device = [3, 3, 4]
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.577800_teammate_pretrain_4/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    #"lbf_run_vanilla_iter_10_ogn_4_ttmax_850000_tmax_1250000_xi_5e-1"
    name = "lbf_run_vanilla_tmax_6250000"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    #cuda_device = [0]
    cuda_device = [1, 1]
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.577800_teammate_pretrain_4/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    #"lbf_run_vanilla_iter_10_ogn_4_ttmax_850000_tmax_1250000_xi_5e-1"
    name = "lbf_run_vanilla_tmax_6250000"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    #cuda_device = [0]
    cuda_device = [4]
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.577800_teammate_pretrain_4/0"
    iterations = 2
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    #"lbf_run_vanilla_iter_10_ogn_4_ttmax_850000_tmax_1250000_xi_5e-1"
    name = "debug_lbf_run_vanilla_tmax_6250000"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = True
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    #cuda_device = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/just_test_gen_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-30_14_49_45.110397_teammate_pretrain_0/0"
    iterations = 2
    z_gen_hyper = False
    #agent = "rnn_ns"
    agent = "rnn_ns"
    teammate_agent = "mlp_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 0
    t_max = 6250000
    once_gen_num = 0
    pretrain_teammate_path =  ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/just_test_gen_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-30_14_49_45.110397_teammate_pretrain_0/0"

program_info = __file__

def one_train(remark, cuda_idx, seed):
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                python src/main.py --config={config}\
                --env-config={env_config}\
                with env_args.time_limit={time_limit}\
                env_args.key={key}\
                --name={name}\
                --learner={learner}\
                --use_encoder={use_encoder}\
                --train_schedule={train_schedule}\
                --test_schedule={test_schedule}\
                --test_function2={test_function2}\
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
                --test_recorder_load_path={test_recorder_load_path} &\
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
        
