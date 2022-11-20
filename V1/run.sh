for SEED in 0 1 2
do
    export CUDA_VISIBLE_DEVICES=1 && python3 src/main.py --config=my_qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-5p-4f-v1" train_schedule="static" test_schedule="fixed_dynamic" name="tr_b-te_fd" seed=$SEED &
    sleep 1s
done