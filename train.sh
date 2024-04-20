##!/bin/bash

# Step 1: generate the training set
python generate_training_set.py --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8

# Step 2: pre-pretrain stage (train the GNN)
python run_pre_pretrain.py --device 0 --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --max_epoch 2000

# Step 3: pre-train stage (train the hypernetwork)
# Grasper
python run_pretrain.py --device 0 --base_rl 'grasper_mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --use_emb_layer --use_act_supervisor
# MT-PSRO
python run_pretrain.py --device 0 --base_rl 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --use_emb_layer --use_act_supervisor
# MT-PSRO-Aug
python run_pretrain.py --device 0 --base_rl 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --use_emb_layer --use_act_supervisor --use_augmentation
# vanilla PSRO
python run_pretrain.py --device 0 --base_rl 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --use_emb_layer

# Step 4: generate the testing set
python generate_testing_set.py --device 0 --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --use_act_supervisor --use_emb_layer --ind_thd_min 0 --ind_thd_max 1 --ood_thd_min 0 --ood_thd_max 1

# Step 5: fine-tuning stage (train BR policies)
# Grasper
python run_psro.py --device 0 --pursuer_runner_type 'grasper_mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer
python run_psro.py --device 0 --pursuer_runner_type 'grasper_mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer --ood_test
# MT-PSRO
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer --ood_test
# MT-PSRO-Aug
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer --use_augmentation
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_act_supervisor --use_emb_layer --use_augmentation --ood_test
# vanilla PSRO
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_emb_layer
python run_psro.py --device 0 --pursuer_runner_type 'mappo' --graph_type 'Grid_Graph' --min_evader_pth_len 6 --edge_probability 0.8 --load_pretrain_model --pretrain_model_iteration 2000 --psro_iteration 3 --train_pursuer_number 10 --use_emb_layer --ood_test