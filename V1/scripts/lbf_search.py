import os
import threading

"""
Do Param Search
"""

if True:
    # 跑12个
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    #name = "lbf_run_iter_10_ogn_4_ttmax_850000_tmax_1250000_lstm_zdim_32_64_xi_5e-1"
    
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    test_function2 = False
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    seeds = [0, 1, 2]
    #cuda_device = [0, 0, 0]
    recorder_load_path = ""
    iterations = 10
    z_gen_hyper = False
    agent = "rnn_ns"
    teammate_agent = "rnn_ns"
    proxy_encoder = "gru_ns"
    team_encoder = "gru"
    use_contrastive_loss = True
    xi =  0.5
    teammate_t_max = 850000
    t_max = 1250000
    once_gen_num = 4
    pretrain_teammate_path =  "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_teammate_path"
    pretrain_enc_path = "pretrain_checkpoint/lbf_6x6_4p3f_coop/pretrain_enc_path"
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed2_Foraging-6x6-4p-3f-coop-v1_2022-12-26_08_31_02.578214_teammate_pretrain_6/0"

    """candidate_contrastive_lambda_1 = []
    candidate_contrastive_lambda_2 = []
    candidate_vi_lambda_1 = []
    candidate_vi_lambda_2 = []
    candidate_kernel = []"""
    proxy_z_dim = 4 # 4
    team_z_dim = 6 # 6
    candidate_param_pair = [(1, 1, .001, .0001, "rbf"), #original
                            (1, 10, .001, .0001, "rbf"),
                            (1, 100, .001, .0001, "rbf"),
                            (1, 1000, .001, .0001, "rbf"),
                            #(1, 1, .0001, .0001, "rbf"),
                            #(1, 10, .0001, .0001, "rbf"),
                            #(1, 100, .0001, .0001, "rbf"),
                            #(1, 1000, .0001, .0001, "rbf"),
                            (1, 1, .001, .0001, "rbf_elemenet_wise"),
                            (1, 10, .001, .0001, "rbf_elemenet_wise"),
                            (1, 100, .001, .0001, "rbf_elemenet_wise"),
                            (1, 1000, .001, .0001, "rbf_elemenet_wise"),]

    cuda_device = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

program_info = __file__

def one_train(remark, cuda_idx, seed, param_pair):
    contrastive_lambda_1, contrastive_lambda_2, vi_lambda_1, vi_lambda_2, kernel = param_pair
    name = f"lbf_search_contralmb_{contrastive_lambda_1}_{contrastive_lambda_2}_vilmd_{vi_lambda_1}_{vi_lambda_2}_kernel_{kernel}"
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                python src/main.py --config={config}\
                --env-config={env_config}\
                with env_args.time_limit={time_limit}\
                env_args.key={key}\
                --contrastive_lambda_1={contrastive_lambda_1}\
                --contrastive_lambda_2={contrastive_lambda_2}\
                --vi_lambda_1={vi_lambda_1}\
                --vi_lambda_2={vi_lambda_2}\
                --kernel={kernel}\
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
    for param_pair, cuda_idx in zip(candidate_param_pair, cuda_device):
        for seed in seeds:
            remark = f"{base_remark}_{cuda_idx}"
            th = threading.Thread(target=one_train, args=(remark, cuda_idx, seed, param_pair))
            th.start()
            threads.append(th)
    
    for th in threads:
        th.join()