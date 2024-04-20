import os
import argparse
import os.path as osp


def get_config():

    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=["rmappo", "mappo"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training")

    # env parameters
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")

    # network parameters
    parser.add_argument("--share_policy", action='store_false', default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false', default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true', default=False, help="Whether to use stacked_frames")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_true', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_true', default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_true', default=False, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true', default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false', default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10, help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_true', default=False, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1.0, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_true', default=False, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_advnorm", action='store_false', default=True, help='use normalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--use_gae", action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_true', default=False, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",  action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, help='use a linear schedule on the learning rate')

    return parser


def get_mtl_model_results_dir(args, iteration):
    save_path = ''
    if args.graph_type == "Grid_Graph":
        save_path = './data/pretrain_models/mappo/grid_{}_probability_{}/pretrain_model'\
            .format(args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = f'./data/pretrain_models/mappo/sg_graph_probability_{args.edge_probability}/pretrain_model'
    elif args.graph_type == 'SY_Graph':
        save_path = f'./data/pretrain_models/mappo/sy_graph/pretrain_model'
    elif args.graph_type == 'SF_Graph':
        save_path = f'./data/pretrain_models/mappo/sf_graph_{args.sf_sw_node_num}/pretrain_model'

    if not osp.exists(save_path):
        os.makedirs(save_path)

    location = "num_gts{}_{}_{}_iter{}_bsize{}_node_feat{}_dnum{}_enum{}_T{}_{}_hdim{}_mep{}"\
        .format(args.num_games, args.num_task, args.num_sample, iteration, args.batch_size, args.node_feat_dim,
                args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon,
                args.hidden_size, args.min_evader_pth_len)

    if args.use_emb_layer:
        location += "_use_el"
    if args.use_augmentation:
        location += "_aug"
    location += f"_gp{args.pool_size}"
    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    return osp.join(save_path, location)


def get_train_utility_results_dir(args):
    save_path = ''
    if args.graph_type == "Grid_Graph":
        save_path = './data/pretrain_models/mappo/grid_{}_probability_{}/utility_record'.format(
            args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = f'./data/pretrain_models/mappo/sg_graph_probability_{args.edge_probability}/utility_record'
    elif args.graph_type == 'SY_Graph':
        save_path = f'./data/pretrain_models/mappo/sy_graph/utility_record'
    elif args.graph_type == 'SF_Graph':
        save_path = f'./data/pretrain_models/mappo/sf_graph_{args.sf_sw_node_num}/utility_record'

    if not osp.exists(save_path):
        os.makedirs(save_path)

    location = "seed{}_num_gts{}_{}_{}_bsize{}_node_feat{}_dnum{}_enum{}_T{}_{}_hdim{}_mep{}" \
        .format(args.seed, args.num_games, args.num_task, args.num_sample, args.batch_size, args.node_feat_dim,
                args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon,
                args.hidden_size, args.min_evader_pth_len)

    if args.use_emb_layer:
        location += "_use_el"
    if args.use_augmentation:
        location += "_aug"
    location += f"_gp{args.pool_size}"
    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)
    location += "_reward.pik"

    return osp.join(save_path, location)


def get_runs_dir(args):
    location = ""
    if args.graph_type == "Grid_Graph":
        location = "mappo_grid_{}_probability_{}"\
            .format(args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        location = f"mappo_sg_graph_probability_{args.edge_probability}"
    elif args.graph_type == 'SY_Graph':
        location = f"mappo_sy_graph"
    elif args.graph_type == 'SF_Graph':
        location = f"mappo_sf_graph_{args.sf_sw_node_num}"
    location += "_seed{}_num_gts{}_{}_{}_bsize{}_node_feat{}_dnum{}_enum{}_T{}_{}_hdim{}_mep{}" \
        .format(args.seed, args.num_games, args.num_task, args.num_sample, args.batch_size, args.node_feat_dim,
                args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon,
                args.hidden_size, args.min_evader_pth_len)

    if args.use_emb_layer:
        location += "_use_el"
    if args.use_augmentation:
        location += "_aug"
    location += f"_gp{args.pool_size}"
    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    return location