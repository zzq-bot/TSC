export CUDA_VISIBLE_DEVICES=0
SEED=$1
#python3 src/main.py --config=my_qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-5p-4f-v1" train_schedule="base" test_schedule="fixed_dynamic" name="tr_s-te_fd" seed=$SEED
python3 src/main.py --config=my_qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-5p-4f-v1" learner='q_learner' use_encoder=False train_schedule="base" test_schedule="fixed_dynamic" name="tr_s-te_fd" seed=$SEED
#python src/main.py --config=my_qmix --env-config=sc2 with env_args.map_name=2s3z seed=0