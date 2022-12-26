import os
import threading

"""
Test the second function: How to train controllable agents
"""

"""if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 50
    key = "Foraging-8x8-4p-4f-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "train"
    name = "test_controllable"
    seeds = [0]
    cuda_device = [0]
    recorder_load_path = "recorder_checkpoint/teammate_pretrain_seed0_Foraging-8x8-4p-4f-v1_2022-11-14_16:15:40.710435_test_crp_0/0"
    #checkpoint_path = ""
    #once_gen_num = 4
    #teammate_t_max = 10000
    iterations = 2 # do not need train controllable agents


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
                --test_function2=True\
                --recorder_load_path={recorder_load_path}\
                --iterations={iterations} &\
                sleep 2s"
    else:
        assert 0
    ret = os.system(cmd)
    if ret != 0:
        print("failure !!!!")
"""
if True:
    config = "my_qmix"
    env_config = "gymma"
    time_limit = 25
    key = "Foraging-6x6-4p-3f-coop-v1"
    learner = 'my_q_learner'
    use_encoder = True
    train_schedule = "train"
    test_schedule = "test"
    # ogn:once_gen_num; ttmax: teammate_t_max; z_dim_x_y: proxy_z-x, team_z-y
    name = "just_test"
    seeds = [0]
    cuda_device = [1]
    recorder_load_path = ""
    test_function2 = True
    iterations = 2
    z_gen_hyper = False
    use_contrastive_loss = True
    proxy_z_dim = 4
    team_z_dim = 8
    xi = .5
    teammate_t_max = 1000
    t_max = 30000
    once_gen_num = 3
    pretrain_teammate_path = ""
    pretrain_enc_path = ""
    test_recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-24_18_26_38.875653_teammate_pretrain_0/0"
    recorder_load_path = "recorder_checkpoint/lbf_6643c_teammate_pretrain_seed0_Foraging-6x6-4p-3f-coop-v1_2022-12-24_18_26_38.875653_teammate_pretrain_0/0"
    agent = "rnn_ns"
    proxy_encoder = "lstm_ns"
    team_encoder = "lstm"
    teammate_agent = "rnn_ns"


program_info = __file__

def one_train(remark, cuda_idx, seed):
    if env_config == "gymma":
        cmd = f"export CUDA_VISIBLE_DEVICES={cuda_idx} &&\
                python3 src/main.py --config={config}\
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
                --proxy_encoder={proxy_encoder}\
                --team_encoder={team_encoder}\
                --use_contrastive_loss={use_contrastive_loss}\
                --proxy_z_dim={proxy_z_dim}\
                --team_z_dim={team_z_dim}\
                --xi={xi}\
                --teammate_t_max={teammate_t_max}\
                --t_max={t_max}\
                --once_gen_num={once_gen_num}\
                --pretrain_enc_path={pretrain_enc_path}\
                --pretrain_teammate_path={pretrain_teammate_path}\
                --recorder_load_path={recorder_load_path}\
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
        
