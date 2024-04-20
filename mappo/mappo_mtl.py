import time
import torch
import sys
from tensorboardX import SummaryWriter
import os
import pickle
from datetime import datetime
import numpy as np
import os.path as osp

from mappo.envs.env import RL_Env
from mappo.runner_shared.env_runner import EnvRunner as Runner
from mappo.config import get_config, get_train_utility_results_dir, get_mtl_model_results_dir, get_runs_dir

from graph_learning.encoder import PreModel
from utils.graph_learning_utils import get_dgl_graph


def mappo_mtl(args):
    if args.graph_type == 'Grid_Graph':
        save_path = 'data/related_files/game_pool/grid_{}_probability_{}'.format(args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = 'data/related_files/game_pool/sg_graph_probability_{}'.format(args.edge_probability)
    elif args.graph_type == 'SY_Graph':
        save_path = 'data/related_files/game_pool/sy_graph'
    elif args.graph_type == 'SF_Graph':
        save_path = 'data/related_files/game_pool/sf_graph_{}'.format(args.sf_sw_node_num)
    else:
        raise ValueError("Unrecognized graph type")
    file_path = osp.join(save_path, 'game_pool_size{}_dnum{}_enum{}_T{}_{}_mep{}.pik'
                          .format(args.pool_size, args.num_defender, args.num_exit, args.min_time_horizon, args.max_time_horizon, args.min_evader_pth_len))
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print('Load game pool ...')
        print(file_path)
        game_pool = pickle.load(open(file_path, 'rb'))['game_pool']
    else:
        raise ValueError('Game pool does not exist.')

    game = np.random.choice(game_pool)

    feat = game.get_graph_info()
    parser = get_config()
    mappo_args = parser.parse_known_args(sys.argv[1:])[0]
    setup_str = f"Emb Layer: {args.use_emb_layer}, Augment: {args.use_augmentation}, Act Supervisor: {args.use_act_supervisor}"
    if args.graph_type == 'Grid_Graph':
        args.node_num = args.row_max_for_state_emb * args.column_max_for_state_emb  # enable varied grid size
    else:
        args.node_num = feat.shape[0]
    args.defender_num = game._defender_num

    env = RL_Env(game)
    runner = Runner(env, mappo_args, args, train_mode='pretrain')

    if args.use_augmentation:
        graph_emb_model = PreModel(feat.shape[1], args.gnn_hidden_dim, args.gnn_output_dim, args.gnn_num_layer, args.gnn_dropout)
        graph_emb_model.to(args.device)
        if args.load_graph_emb_model:
            print('Load pretrained GNN model ...')
            pretrain_graph_model_file = f"data/pretrain_models/graph_learning/checkpoint_epoch{args.max_epoch}_type_{args.graph_type}_" \
                                        f"ep{args.edge_probability}_gp{args.pool_size}_layer{args.gnn_num_layer}_" \
                                        f"hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.num_defender}_" \
                                        f"enum{args.num_exit}_mep{args.min_evader_pth_len}.pt"
            print(pretrain_graph_model_file)
            graph_emb_model.load(torch.load(pretrain_graph_model_file))
        graph_emb_model.eval()
        with torch.no_grad():
            hg = get_dgl_graph(game)
            hg = hg.to(args.device)
            feat = hg.ndata["attr"]
            _, pooled_node_emb = graph_emb_model.embed(hg, feat)
            pooled_node_emb = pooled_node_emb.cpu().numpy()
            pooled_node_embs = np.array([pooled_node_emb for _ in range(runner.num_defender)])
    else:
        pooled_node_embs = None

    T = [0] * args.max_time_horizon_for_state_emb
    T[game._time_horizon] = 1
    Ts = np.array([T for _ in range(runner.num_defender)])  # shape = (n_agent, max_time_horizon_for_state_emb)

    time_list, reward_list, aloss_list, vloss_list, itera_list = [], [], [], [], []
    start_iter = 0
    writer = SummaryWriter(comment=get_runs_dir(args))

    if args.checkpoint > 0:
        adaption_reward = 0
        episode_count = 0
        start_iter = args.checkpoint
        actor_checkpoint = get_mtl_model_results_dir(args, args.checkpoint) + "_actor.pt"
        critic_checkpoint = get_mtl_model_results_dir(args, args.checkpoint) + "_critic.pt"
        result_pth = get_train_utility_results_dir(args)
        print("Load Checkpoint {} ......".format(args.checkpoint))
        runner.load_checkpoint(actor_checkpoint, critic_checkpoint)
        if os.path.exists(result_pth) and os.path.getsize(result_pth) > 0:
            data = pickle.load(open(result_pth, 'rb'))
            reward_list, time_list, aloss_list, vloss_list, itera_list = data['reward_list'], data['time_list'], data['aloss_list'], data['vloss_list'], data['itera_list']
            if writer is not None:
                for i in range(len(reward_list)):
                    writer.add_scalar('train_reward_', reward_list[i], itera_list[i])
        if args.use_augmentation:
            with torch.no_grad():
                hg = get_dgl_graph(game)
                hg = hg.to(args.device)
                feat = hg.ndata["attr"]
                _, pooled_node_emb = graph_emb_model.embed(hg, feat)
                pooled_node_emb = pooled_node_emb.cpu().numpy()
                pooled_node_embs = np.array([pooled_node_emb for _ in range(runner.num_defender)])
        else:
            pooled_node_embs = None
        T = [0] * args.max_time_horizon_for_state_emb
        T[game._time_horizon] = 1
        Ts = np.array([T for _ in range(runner.num_defender)])  # shape = (n_agent, max_time_horizon_for_state_emb)

    # train
    min_update_episodes = args.num_games * args.num_task * args.num_sample
    update_game_freq = args.num_task * args.num_sample
    start_ = datetime.now().replace(microsecond=0)
    start_time = time.time()

    for iteration in range(start_iter, args.num_iterations):
        start_time_iter = time.time()
        runner.trainer.policy.actor.eval()
        runner.trainer.policy.critic.eval()

        # sample a new game, get the node_feat and time_horizon of the new game
        if iteration % update_game_freq == 0:
            adaption_reward = 0
            episode_count = 0
            game = np.random.choice(game_pool)
            runner.reset(game)
            if args.use_augmentation:
                with torch.no_grad():
                    hg = get_dgl_graph(game)
                    hg = hg.to(args.device)
                    feat = hg.ndata["attr"]
                    _, pooled_node_emb = graph_emb_model.embed(hg, feat)
                    pooled_node_emb = pooled_node_emb.cpu().numpy()
                    pooled_node_embs = np.array([pooled_node_emb for _ in range(runner.num_defender)])
            else:
                pooled_node_embs = None
            T = [0] * args.max_time_horizon_for_state_emb
            T[game._time_horizon] = 1
            Ts = np.array([T for _ in range(runner.num_defender)])  # shape = (n_agent, max_time_horizon_for_state_emb)

        # generate a new attacker strategy
        if update_game_freq > 1 and iteration % args.num_sample == 0:
            runner.env.initialize_attacker_strategy()

        # play with this new attacker strategy and store the data
        episode_reward = runner.run_one_episode(pooled_node_embs, Ts, use_act_supervisor=args.use_act_supervisor)
        adaption_reward += episode_reward
        episode_count += 1

        # update the policy
        min_update_eps = min_update_episodes if min_update_episodes > 1 else args.batch_size
        if args.checkpoint == 0:
            if (iteration + 1) >= min_update_eps and (iteration + 1) % min_update_eps == 0:
                train_infos = runner.train()
                aloss_list.append(train_infos['value_loss'])
                vloss_list.append(train_infos['policy_loss'])
        else:
            if (iteration + 1) >= args.checkpoint + min_update_eps and (iteration + 1) % min_update_eps == 0:
                train_infos = runner.train()
                aloss_list.append(train_infos['value_loss'])
                vloss_list.append(train_infos['policy_loss'])

        print_every = update_game_freq * 10 if update_game_freq > 1 else 1000
        if (iteration + 1) % print_every == 0 or iteration + 1 == args.num_iterations:
            print('{}, {}, Iteration: {}/{}, Train Reward: {:.6f}, MEP: {}, Time: {:.6f}'
                  .format(args.base_rl, setup_str, iteration + 1, args.num_iterations, adaption_reward / episode_count,
                          args.min_evader_pth_len, time.time() - start_time_iter))

        record_every = update_game_freq if update_game_freq > 1 else 1000
        if (iteration + 1) % record_every == 0 or iteration + 1 == args.num_iterations:
            itera_list.append(iteration)
            if writer is not None:
                writer.add_scalar('train_reward_', adaption_reward / episode_count, iteration)

            end_time = time.time()
            time_list.append(end_time - start_time)
            reward_list.append(adaption_reward / episode_count)

        save_res_every = (100 * update_game_freq) if update_game_freq > 1 else 100000
        if iteration == 0 or (iteration + 1) % save_res_every == 0 or iteration + 1 == args.num_iterations:
            pickle.dump({'reward_list': reward_list, 'time_list': time_list, 'aloss_list': aloss_list,
                         'vloss_list': vloss_list, 'itera_list': itera_list},
                        open(get_train_utility_results_dir(args), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.num_iterations:
            runner.save(get_mtl_model_results_dir(args, iteration + 1))

    print("============================================================================================")
    end_ = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_)
    print("Finished training at (GMT) : ", end_)
    print("Total training time  : ", end_ - start_)
    print("============================================================================================")