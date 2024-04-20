import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling

from graph_learning.gcn import GCN


class Head(nn.Module):
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev, init_method):
        super(Head, self).__init__()
        h_layer = latent_dim
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
        self.W1 = nn.Linear(h_layer, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(h_layer, output_dim_out)
        self.init_method = init_method
        self.init_layers(sttdev)

    def forward(self, x):
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        return w, b

    def init_layers(self, stddev):
        if self.init_method == 'unif':
            torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
            torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
            torch.nn.init.zeros_(self.W1.bias)
            torch.nn.init.zeros_(self.b1.bias)
        elif self.init_method == 'xavi':
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.b1.weight)
            torch.nn.init.constant_(self.W1.bias, 0)
            torch.nn.init.constant_(self.b1.bias, 0)
        elif self.init_method == 'kaim':
            nn.init.kaiming_uniform_(self.W1.weight.data, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.b1.weight.data, nonlinearity='relu')
            torch.nn.init.constant_(self.W1.bias, 0)
            torch.nn.init.constant_(self.b1.bias, 0)


class Meta_Embadding(nn.Module):
    def __init__(self, meta_dim, hidden_dim=128, z_dim=128):
        super(Meta_Embadding, self).__init__()
        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.init_layers()

    def forward(self, meta_v, frozen=False):
        if frozen:
            with torch.no_grad():
                z = self.hyper(meta_v).view(-1, self.z_dim)
        else:
            z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork(nn.Module):
    def __init__(self, obs_cent_obs_dim, hyper_input_dim, output_dim, device, hyper_hidden_dim=128, z_dim=128,
                 dynamic_hidden_dim=128, use_augmentation=True, head_init_method='kaim'):
        super(HyperNetwork, self).__init__()
        self.use_augmentation = use_augmentation
        self.hyper = Meta_Embadding(hyper_input_dim, hyper_hidden_dim, z_dim)
        self.linear1 = Head(z_dim, obs_cent_obs_dim, dynamic_hidden_dim, sttdev=0.05, init_method=head_init_method)
        self.linear2 = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008, init_method=head_init_method)
        self.linear3 = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001, init_method=head_init_method)
        self.device = device

    def forward(self, pooled_node_emb, T, base_v, batch=False):
        meta_v = torch.cat([pooled_node_emb, T], dim=1)
        z = self.hyper(meta_v)
        w1, b1 = self.linear1(z)
        w2, b2 = self.linear2(z)
        w3, b3 = self.linear3(z)
        if self.use_augmentation:
            base_v = torch.cat([base_v, pooled_node_emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out.squeeze(0)

    def get_weight(self, pooled_node_emb, T):
        with torch.no_grad():
            meta_v = torch.cat([pooled_node_emb, T], dim=1)
            z = self.hyper(meta_v)
            w1, b1 = self.linear1(z)
            w2, b2 = self.linear2(z)
            w3, b3 = self.linear3(z)
        return [w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)], \
            [b1.squeeze(0).squeeze(1), b2.squeeze(0).squeeze(1), b3.squeeze(0).squeeze(1)]


class Actor_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, action_dim, node_num, defender_num, device, hyper_hidden_dim=128, z_dim=128,
                 dynamic_hidden_dim=128, head_init_method='kaim'):
        super(Actor_With_Emb_Layer, self).__init__()
        self.args = args
        if args.use_augmentation:
            policy_input_dim = args.state_emb_dim * (defender_num + 3) + args.gnn_output_dim
        else:
            policy_input_dim = args.state_emb_dim * (defender_num + 3)
        self.hyper = Meta_Embadding(hyper_input_dim, hyper_hidden_dim, z_dim)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.agent_id_emb_layer = nn.Embedding(defender_num, args.state_emb_dim)
        self.linear1 = Head(z_dim, policy_input_dim, dynamic_hidden_dim, sttdev=0.05, init_method=head_init_method)
        self.linear2 = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008, init_method=head_init_method)
        self.linear3 = Head(z_dim, dynamic_hidden_dim, action_dim, sttdev=0.001, init_method=head_init_method)
        self.device = device

    def forward(self, pooled_node_emb, T, base_v, batch=False):
        meta_v = torch.cat([pooled_node_emb, T], dim=1)
        z = self.hyper(meta_v)
        w1, b1 = self.linear1(z)
        w2, b2 = self.linear2(z)
        w3, b3 = self.linear3(z)
        node_idx = base_v[:, :-2] if batch else base_v[:-2].unsqueeze(0)
        time_idx = base_v[:, -2] if batch else base_v[-2].unsqueeze(0)
        agent_id = base_v[:, -1] if batch else base_v[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_emb], dim=1)
        out = F.relu(torch.bmm(w1, state_emb.unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out.squeeze(0)

    def get_weight(self, pooled_node_emb, T):
        with torch.no_grad():
            meta_v = torch.cat([pooled_node_emb, T], dim=1)
            z = self.hyper(meta_v)
            w1, b1 = self.linear1(z)
            w2, b2 = self.linear2(z)
            w3, b3 = self.linear3(z)
        return [w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)], \
            [b1.squeeze(0).squeeze(1), b2.squeeze(0).squeeze(1), b3.squeeze(0).squeeze(1)]

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)


class Critic_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, node_num, defender_num, device, hyper_hidden_dim=128, z_dim=128,
                 dynamic_hidden_dim=128, head_init_method='kaim'):
        super(Critic_With_Emb_Layer, self).__init__()
        self.args = args
        if args.use_augmentation:
            value_input_dim = args.state_emb_dim * (defender_num + 2) + args.gnn_output_dim
        else:
            value_input_dim = args.state_emb_dim * (defender_num + 2)
        self.hyper = Meta_Embadding(hyper_input_dim, hyper_hidden_dim, z_dim)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.linear1 = Head(z_dim, value_input_dim, dynamic_hidden_dim, sttdev=0.05, init_method=head_init_method)
        self.linear2 = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008, init_method=head_init_method)
        self.linear3 = Head(z_dim, dynamic_hidden_dim, 1, sttdev=0.001, init_method=head_init_method)
        self.device = device

    def forward(self, pooled_node_emb, T, base_v, batch=False):
        meta_v = torch.cat([pooled_node_emb, T], dim=1)
        z = self.hyper(meta_v)
        w1, b1 = self.linear1(z)
        w2, b2 = self.linear2(z)
        w3, b3 = self.linear3(z)
        node_idx = base_v[:, :-1] if batch else base_v[:-1].unsqueeze(0)
        time_idx = base_v[:, -1] if batch else base_v[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        state_emb = torch.cat([node_idx_emb, time_idx_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_emb], dim=1)
        out = F.relu(torch.bmm(w1, state_emb.unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out.squeeze(0)

    def get_weight(self, pooled_node_emb, T):
        with torch.no_grad():
            meta_v = torch.cat([pooled_node_emb, T], dim=1)
            z = self.hyper(meta_v)
            w1, b1 = self.linear1(z)
            w2, b2 = self.linear2(z)
            w3, b3 = self.linear3(z)
        return [w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)], \
            [b1.squeeze(0).squeeze(1), b2.squeeze(0).squeeze(1), b3.squeeze(0).squeeze(1)]

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)


class Actor_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, action_dim, node_num, defender_num, device, hyper_hidden_dim=128, z_dim=128,
                 dynamic_hidden_dim=128, head_init_method='kaim'):
        super(Actor_With_Emb_Layer_Scratch, self).__init__()
        self.args = args
        if args.use_augmentation:
            policy_input_dim = args.state_emb_dim * (defender_num + 3) + args.gnn_output_dim
        else:
            policy_input_dim = args.state_emb_dim * (defender_num + 3)
        self.encoder = GCN(
            in_dim=args.feat_dim,
            num_hidden=args.gnn_hidden_dim,
            out_dim=args.gnn_output_dim,
            num_layers=args.gnn_num_layer,
            dropout=args.gnn_dropout,
            encoding=True
        )
        self.hyper = Meta_Embadding(args.gnn_output_dim + args.max_time_horizon_for_state_emb, hyper_hidden_dim, z_dim)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.agent_id_emb_layer = nn.Embedding(defender_num, args.state_emb_dim)
        self.linear1 = Head(z_dim, policy_input_dim, dynamic_hidden_dim, sttdev=0.05, init_method=head_init_method)
        self.linear2 = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008, init_method=head_init_method)
        self.linear3 = Head(z_dim, dynamic_hidden_dim, action_dim, sttdev=0.001, init_method=head_init_method)
        self.device = device
        self.pooler = AvgPooling()

    def forward(self, hgs, T, base_v, batch=False):
        feats = hgs.ndata["attr"]
        rep = self.encoder(hgs, feats)
        pooled_node_emb = self.pooler(hgs, rep)
        if not batch:
            pooled_node_emb = pooled_node_emb.squeeze(0)
        meta_v = torch.cat([pooled_node_emb, T], dim=1)
        z = self.hyper(meta_v)
        w1, b1 = self.linear1(z)
        w2, b2 = self.linear2(z)
        w3, b3 = self.linear3(z)
        node_idx = base_v[:, :-2] if batch else base_v[:-2].unsqueeze(0)
        time_idx = base_v[:, -2] if batch else base_v[-2].unsqueeze(0)
        agent_id = base_v[:, -1] if batch else base_v[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_emb], dim=1)
        out = F.relu(torch.bmm(w1, state_emb.unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out.squeeze(0)

    def get_weight(self, hgs,  T):
        with torch.no_grad():
            feats = hgs.ndata["attr"]
            rep = self.encoder(hgs, feats)
            pooled_node_emb = self.pooler(hgs, rep)
            pooled_node_emb = pooled_node_emb
            meta_v = torch.cat([pooled_node_emb, T], dim=1)
            z = self.hyper(meta_v)
            w1, b1 = self.linear1(z)
            w2, b2 = self.linear2(z)
            w3, b3 = self.linear3(z)
        return [w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)], \
            [b1.squeeze(0).squeeze(1), b2.squeeze(0).squeeze(1), b3.squeeze(0).squeeze(1)], rep.cpu(), pooled_node_emb.squeeze(0).cpu()

    def get_node_emb_from_gnn(self, hgs):
        with torch.no_grad():
            feats = hgs.ndata["attr"]
            rep = self.encoder(hgs, feats)
            pooled_node_emb = self.pooler(hgs, rep)
            pooled_node_emb = pooled_node_emb.squeeze(0)
        return rep, pooled_node_emb

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)


class Critic_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, node_num, defender_num, device, hyper_hidden_dim=128, z_dim=128,
                 dynamic_hidden_dim=128, head_init_method='kaim'):
        super(Critic_With_Emb_Layer_Scratch, self).__init__()
        self.args = args
        if args.use_augmentation:
            value_input_dim = args.state_emb_dim * (defender_num + 2) + args.gnn_output_dim
        else:
            value_input_dim = args.state_emb_dim * (defender_num + 2)
        self.encoder = GCN(
            in_dim=args.feat_dim,
            num_hidden=args.gnn_hidden_dim,
            out_dim=args.gnn_output_dim,
            num_layers=args.gnn_num_layer,
            dropout=args.gnn_dropout,
            encoding=True
        )
        self.hyper = Meta_Embadding(args.gnn_output_dim + args.max_time_horizon_for_state_emb, hyper_hidden_dim, z_dim)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.linear1 = Head(z_dim, value_input_dim, dynamic_hidden_dim, sttdev=0.05, init_method=head_init_method)
        self.linear2 = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008, init_method=head_init_method)
        self.linear3 = Head(z_dim, dynamic_hidden_dim, 1, sttdev=0.001, init_method=head_init_method)
        self.device = device
        self.pooler = AvgPooling()

    def forward(self, hgs, T, base_v, batch=False):
        feats = hgs.ndata["attr"]
        rep = self.encoder(hgs, feats)
        pooled_node_emb = self.pooler(hgs, rep)
        if not batch:
            pooled_node_emb = pooled_node_emb.squeeze(0)
        meta_v = torch.cat([pooled_node_emb, T], dim=1)
        z = self.hyper(meta_v)
        w1, b1 = self.linear1(z)
        w2, b2 = self.linear2(z)
        w3, b3 = self.linear3(z)
        node_idx = base_v[:, :-1] if batch else base_v[:-1].unsqueeze(0)
        time_idx = base_v[:, -1] if batch else base_v[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        state_emb = torch.cat([node_idx_emb, time_idx_emb], dim=1)
        if self.args.use_augmentation:
            state_emb = torch.cat([state_emb, pooled_node_emb], dim=1)
        out = F.relu(torch.bmm(w1, state_emb.unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out.squeeze(0)

    def get_weight(self, hgs, T):
        with torch.no_grad():
            feats = hgs.ndata["attr"]
            rep = self.encoder(hgs, feats)
            pooled_node_emb = self.pooler(hgs, rep)
            pooled_node_emb = pooled_node_emb
            meta_v = torch.cat([pooled_node_emb, T], dim=1)
            z = self.hyper(meta_v)
            w1, b1 = self.linear1(z)
            w2, b2 = self.linear2(z)
            w3, b3 = self.linear3(z)
        return [w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)], \
            [b1.squeeze(0).squeeze(1), b2.squeeze(0).squeeze(1), b3.squeeze(0).squeeze(1)], rep.cpu(), pooled_node_emb.squeeze(0).cpu()

    def get_node_emb_from_gnn(self, hgs):
        with torch.no_grad():
            feats = hgs.ndata["attr"]
            rep = self.encoder(hgs, feats)
            pooled_node_emb = self.pooler(hgs, rep)
            pooled_node_emb = pooled_node_emb.squeeze(0)
        return rep, pooled_node_emb

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)