import dgl
import torch
import numpy as np
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os.path as osp
import os
from tensorboardX import SummaryWriter
import time
import pickle

from graph_learning.encoder import PreModel
from utils.graph_learning_utils import (
    create_optimizer,
    set_random_seed,
    get_dgl_graph
)

def collate_fn(batch):
    graphs = [x for x in batch]
    batch_g = dgl.batch(graphs)
    return batch_g


def graph_pretrain(args):
    save_path = './data/pretrain_models/graph_learning'
    if not osp.exists(save_path):
        os.makedirs(save_path)

    lr = 0.00015
    weight_decay = 1e-5
    max_epoch = args.max_epoch

    start_time = time.time()
    if args.graph_type == 'Grid_Graph':
        pth = f'data/related_files/game_pool/grid_{args.row * args.column}_probability_{args.edge_probability}'
    elif args.graph_type == 'SG_Graph':
        pth = f'data/related_files/game_pool/sg_graph_probability_{args.edge_probability}'
    elif args.graph_type == 'SY_Graph':
        pth = f'data/related_files/game_pool/sy_graph'
    elif args.graph_type == 'SF_Graph':
        pth = f'data/related_files/game_pool/sf_graph_{args.sf_sw_node_num}'
    else:
        raise ValueError(f"Unknown graph type: {args.graph_type}.")
    file_path = osp.join(pth, f'game_pool_size{args.pool_size}_dnum{args.num_defender}_enum{args.num_exit}_'
                              f'T{args.min_time_horizon}_{args.max_time_horizon}_mep{args.min_evader_pth_len}.pik')
    print('Load game pool ...')
    game_pool = pickle.load(open(file_path, 'rb'))['game_pool']
    graphs = []
    for game in game_pool:
        hg = get_dgl_graph(game)
        graphs.append(hg)
    game_pool_str = f"_gp{args.pool_size}"

    node_feat_dim = hg.ndata['attr'].shape[1]
    print(f"******** # Num Graphs: {len(graphs)}, # Num Feat: {node_feat_dim} ********")
    writer = SummaryWriter(comment=f'graph_feat_learning_type_{args.graph_type}_ep{args.edge_probability}{game_pool_str}_'
                                   f'layer{args.gnn_num_layer}_hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.num_defender}_'
                                   f'enum{args.num_exit}_mep{args.min_evader_pth_len}')

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=32, pin_memory=True)

    set_random_seed(args.seed)
    model = PreModel(node_feat_dim, args.gnn_hidden_dim, args.gnn_output_dim, args.gnn_num_layer, args.gnn_dropout)
    model.to(args.device)

    optimizer = create_optimizer("adam", model, lr, weight_decay)
    scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    epoch_iter = tqdm(range(max_epoch))
    train_loss = []
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch_g in train_loader:
            batch_g = batch_g.to(args.device)
            feat = batch_g.ndata["attr"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        mean_loss = np.mean(loss_list)
        train_loss.append(mean_loss)
        epoch_iter.set_description(f"Epoch {epoch + 1} | train_loss: {mean_loss:.4f}")
        writer.add_scalar('gnn_train_loss', mean_loss, epoch + 1)
        if (epoch + 1) % 200 == 0:
            model.save(save_path + f"/checkpoint_epoch{epoch + 1}_type_{args.graph_type}_ep{args.edge_probability}{game_pool_str}_layer{args.gnn_num_layer}_"
                                   f"hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.num_defender}_enum{args.num_exit}_"
                                   f"mep{args.min_evader_pth_len}.pt")
    end_time = time.time()
    train_time = end_time - start_time
    pickle.dump({'train_time': train_time, 'train_loss': train_loss},
                open(save_path + f'/train_record_type_{args.graph_type}_ep{args.edge_probability}{game_pool_str}_layer{args.gnn_num_layer}_'
                                 f'hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.num_defender}_enum{args.num_exit}_'
                                 f'mep{args.min_evader_pth_len}.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)