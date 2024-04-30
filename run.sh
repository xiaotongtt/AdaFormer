

# train
#python main.py --GPU '0' --model adaformer --trainer trainer_adaformer --loss '1*L1+1*UNCERTY_SIGMOID' --save results --epochs 1000 --save_results --embed_dim 64  --query_num 16 --n_resblocks 16 --pspl_module "PSPL" --pspl_depths 6 --threshold_eps 0.05

# test
python main.py --GPU '0' --model  adaformer --trainer trainer_adaformer  --save results  --test_only --pre_train "./pretrained_model/model/model_x4.pt" --data_test test2k --n_resblocks 16 --embed_dim 64 --query_num 16 --threshold_eps 0.05
# +test4k+test8k
# Set5+Set14+B100+Urban100