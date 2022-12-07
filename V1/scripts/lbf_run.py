import os
import threading

"""
Run the whole process
"""

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    #name = "run_iter_10_ogn_5_ttmax_550000_tmax_1050000_zdim_32_64_xi_1"
    name = "just_test"
    seeds = [0]
    cuda_device = [2]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  1
    teammate_t_max = 50000
    t_max = 1050000
    once_gen_num = 1
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"



if False:
    # DONT USE THIS
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "run_iter_10_ogn_5_ttmax_550000_tmax_1250000_zdim_32_64_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = .5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"

if False:
    # DONT USE THIS
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "run_iter_10_ogn_5_ttmax_550000_tmax_1250000_zdim_4_8_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [3, 3, 3]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = True
    agent = "mlp_gen_ns"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8
    xi =  0.5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = False
    train_schedule = "train"
    test_schedule = "test"
    name = "vanilla_qmix_iter_10_ogn_5_ttmax_550000_tmax_1250000_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [4, 4, 4]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10 # do not need train controllable agents
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = False
    proxy_z_dim = 0
    team_z_dim = 0
    xi = 0.5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"


if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # cl for contrastive loss
    name = "ablation_wo_cl_iter_10_ogn_5_ttmax_550000_tmax_1250000_zdim_32_64_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [0, 0, 0]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10 # do not need train controllable agents
    z_gen_hyper = True
    agent = "mlp_ns"
    use_contrastive_loss = False
    proxy_z_dim = 32
    team_z_dim = 64
    xi =  0.5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "run_iter_10_ogn_5_mlp_ns_ttmax_550000_tmax_1250000_zdim_4_8_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [1, 1, 1]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8
    xi = .5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"

if False:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "run_iter_10_ogn_5_mlp_ns_ttmax_550000_tmax_1250000_zdim_32_64_xi_0.5"
    seeds = [0, 1, 2]
    cuda_device = [3, 3, 3]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 32
    team_z_dim = 64
    xi = .5
    teammate_t_max = 550000
    t_max = 1250000
    once_gen_num = 5
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"

if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-6x6-4p-5f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "just_test"
    seeds = [0]
    cuda_device = [1]
    recorder_load_path = ""
    test_function2 = False
    iterations = 10
    z_gen_hyper = False
    agent = "mlp_ns"
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8
    xi = .5
    teammate_t_max = 1000
    t_max = 5000
    once_gen_num = 3
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p5f/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-6x6-4p-5f-v1_2022-11-15_11_32_02.636204_teammate_pretrain_4/0"

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
        
