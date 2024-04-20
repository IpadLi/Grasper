import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mappo.utils.util import kaiming_uniform


class R_Actor_With_Emb_Layer(nn.Module):
    def __init__(self, args, action_dim, node_num, defender_num, device=torch.device("cpu")):
        super(R_Actor_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hidden_size
        if args.use_augmentation:
            policy_input_dim = args.state_emb_dim * (defender_num + 3) + args.gnn_output_dim + args.max_time_horizon_for_state_emb
        else:
            policy_input_dim = args.state_emb_dim * (defender_num + 3)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.agent_id_emb_layer = nn.Embedding(defender_num, args.state_emb_dim)
        self.linear1 = nn.Linear(policy_input_dim, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear3 = nn.Linear(args.hidden_size, action_dim)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_emb=None, T=None, batch=False):
        node_idx = states[:, :-2] if batch else states[:-2].unsqueeze(0)
        time_idx = states[:, -2] if batch else states[-2].unsqueeze(0)
        agent_id = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        if pooled_node_emb is None:
            state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        else:
            state_emb = torch.cat([pooled_node_emb, T, node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, states, action, pooled_node_emb=None, T=None, batch=False):
        node_idx = states[:, :-2] if batch else states[:-2].unsqueeze(0)
        time_idx = states[:, -2] if batch else states[-2].unsqueeze(0)
        agent_id = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        agent_id_emb = self.agent_id_emb_layer(agent_id)
        if pooled_node_emb is None:
            state_emb = torch.cat([node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        else:
            state_emb = torch.cat([pooled_node_emb, T, node_idx_emb, time_idx_emb, agent_id_emb], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)


class R_Critic_With_Emb_Layer(nn.Module):
    def __init__(self, args, node_num, defender_num,  device=torch.device("cpu")):
        super(R_Critic_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hidden_size
        if args.use_augmentation:
            critic_input_dim = args.state_emb_dim * (defender_num + 2) + args.gnn_output_dim + args.max_time_horizon_for_state_emb
        else:
            critic_input_dim = args.state_emb_dim * (defender_num + 2)
        self.node_idx_emb_layer = nn.Embedding(node_num + 1, args.state_emb_dim)
        self.time_idx_emb_layer = nn.Embedding(args.max_time_horizon_for_state_emb, args.state_emb_dim)
        self.linear1 = nn.Linear(critic_input_dim, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear3 = nn.Linear(args.hidden_size, 1)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_emb=None, T=None, batch=False):
        node_idx = states[:, :-1] if batch else states[:-1].unsqueeze(0)
        time_idx = states[:, -1] if batch else states[-1].unsqueeze(0)
        batch_n = node_idx.shape[0]
        node_idx_emb = self.node_idx_emb_layer(node_idx).view(batch_n, -1)
        time_idx_emb = self.time_idx_emb_layer(time_idx)
        if pooled_node_emb is None:
            state_emb = torch.cat([node_idx_emb, time_idx_emb], dim=1)
        else:
            state_emb = torch.cat([pooled_node_emb, T, node_idx_emb, time_idx_emb], dim=1)
        x = F.relu(self.linear1(state_emb))
        x = F.relu(self.linear2(x))
        values = self.linear3(x)
        return values

    def get_node_emb_from_emb_layer(self, node_idx):
        with torch.no_grad():
            return self.node_idx_emb_layer(node_idx)


class R_Actor(nn.Module):
    def __init__(self, args, state_dim, action_dim, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        self.linear1 = nn.Linear(state_dim, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear3 = nn.Linear(args.hidden_size, action_dim)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_emb=None, T=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, states, action, pooled_node_emb=None, T=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        actor_features = self.linear3(x)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()


class R_Critic(nn.Module):
    def __init__(self, args, state_dim, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        self.linear1 = nn.Linear(state_dim, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear3 = nn.Linear(args.hidden_size, 1)
        self.apply(kaiming_uniform)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, states, pooled_node_emb=None, T=None, batch=False):
        x = F.relu(self.linear1(states))
        x = F.relu(self.linear2(x))
        values = self.linear3(x)
        return values