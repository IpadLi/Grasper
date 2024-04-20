import os
import os.path as osp


def get_result_dir(args):
    save_path = ''
    if args.graph_type == "Grid_Graph":
        save_path = './data/psro_results/{}/grid_{}_probability_{}'.format(args.pursuer_runner_type, args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = f'./data/psro_results/{args.pursuer_runner_type}/sg_graph_probability_{args.edge_probability}'
    elif args.graph_type == 'SF_Graph':
        save_path = f'./data/psro_results/{args.pursuer_runner_type}/sf_graph_{args.sf_sw_node_num}'
    elif args.graph_type == 'SY_Graph':
        save_path = f'./data/psro_results/{args.pursuer_runner_type}/sy_graph'
    if not osp.exists(save_path):
        os.makedirs(save_path)

    if args.pursuer_runner_type == 'mappo':
        location = "seed{}_num_gts{}_{}_{}_bsize{}_node_feat{}_dnum{}_enum{}_T{}_{}_hdim{}_mep{}" \
            .format(args.seed, args.num_games, args.num_task, args.num_sample, args.batch_size, args.node_feat_dim,
                    args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon, args.hidden_size, args.min_evader_pth_len)
    else:
        location = "seed{}_num_gts{}_{}_{}_bsize{}_node_feat{}_gnn{}_{}_{}_dnum{}_enum{}_T{}_{}_mep{}" \
            .format(args.seed, args.num_games, args.num_task, args.num_sample, args.batch_size, args.node_feat_dim,
                    args.gnn_num_layer, args.gnn_hidden_dim, args.gnn_output_dim, args.num_defender, args.num_exit,
                    args.min_time_horizon, args.max_time_horizon, args.min_evader_pth_len)

    location += "_evad{}{}".format(args.train_evader_number, "_purs{}_psro{}".format(args.train_pursuer_number, args.psro_iteration))

    if args.load_pretrain_model:
        location += "_ckt_{}".format(args.pretrain_model_iteration)
    else:
        location += "_ckt_no"

    if args.pursuer_runner_type == 'mappo':
        if args.use_emb_layer:
            location += "_use_el"
        if args.use_augmentation:
            location += "_aug"
        location += f"_gp{args.pool_size}"
        if args.ood_test:
            location += f"_ood"
        else:
            location += f"_ind"
    else:
        if args.use_end_to_end:
            if args.use_emb_layer:
                location += "_e2e_use_el"
            if args.use_augmentation:
                location += "_e2e_aug"
            location += f"_gp{args.pool_size}"
            if args.ood_test:
                location += f"_ood"
            else:
                location += f"_ind"
        else:
            if args.use_emb_layer:
                location += "_use_el"
                if args.use_augmentation:
                    location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            location += f"_gp{args.pool_size}"
            if args.ood_test:
                location += f"_ood"
            else:
                location += f"_ind"

    if args.ood_test:
        location += "_thd{}_{}".format(args.ood_thd_min, args.ood_thd_max)
    else:
        location += "_thd{}_{}".format(args.ind_thd_min, args.ind_thd_max)

    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_max, args.act_sup_coef_decay)

    location += ".pik"

    return osp.join(save_path, location)