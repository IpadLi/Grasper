import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from grasper_mappo.algorithms.algorithm.hyper_nets import (
    HyperNetwork,
    Actor_With_Emb_Layer,
    Critic_With_Emb_Layer,
    Actor_With_Emb_Layer_Scratch,
    Critic_With_Emb_Layer_Scratch
)

class H_Actor(nn.Module):
    def __init__(self, args, obs_dim, hyper_input_dim, action_dim, device=torch.device("cpu")):
        super(H_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = HyperNetwork(obs_dim, hyper_input_dim, action_dim, device, dynamic_hidden_dim=self.hidden_size,
                                 use_augmentation=args.use_augmentation, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, obs, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, pooled_node_emb, T, obs, action, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()


class H_Critic(nn.Module):
    def __init__(self, args, cent_obs_dim, hyper_input_dim, device=torch.device("cpu")):
        super(H_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = HyperNetwork(cent_obs_dim, hyper_input_dim, 1, device, dynamic_hidden_dim=self.hidden_size,
                                 use_augmentation=args.use_augmentation, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.base(pooled_node_emb, T, cent_obs, batch=batch)
        return values


class H_Actor_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, action_dim, device=torch.device("cpu")):
        super(H_Actor_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = Actor_With_Emb_Layer(args, hyper_input_dim, action_dim, args.node_num, args.defender_num, device,
                                         dynamic_hidden_dim=self.hidden_size, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, obs, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, pooled_node_emb, T, obs, action, batch=False):
        actor_features = self.base(pooled_node_emb, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()


class H_Critic_With_Emb_Layer(nn.Module):
    def __init__(self, args, hyper_input_dim, device=torch.device("cpu")):
        super(H_Critic_With_Emb_Layer, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = Critic_With_Emb_Layer(args, hyper_input_dim, args.node_num, args.defender_num, device,
                                          dynamic_hidden_dim=self.hidden_size, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.base(pooled_node_emb, T, cent_obs, batch=batch)
        return values


class H_Actor_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, action_dim, device=torch.device("cpu")):
        super(H_Actor_With_Emb_Layer_Scratch, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = Actor_With_Emb_Layer_Scratch(args, action_dim, args.node_num, args.defender_num, device,
                                                 dynamic_hidden_dim=self.hidden_size, head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, hgs, T, obs, batch=False):
        actor_features = self.base(hgs, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions)
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, hgs, T, obs, action, batch=False):
        actor_features = self.base(hgs, T, obs, batch=batch)
        probs = F.softmax(actor_features, dim=-1)
        m = Categorical(probs)
        action_log_probs = m.log_prob(action.squeeze(1))
        dist_entropy = m.entropy().mean()
        return action_log_probs.unsqueeze(1), dist_entropy, (probs + 1e-8).log()


class H_Critic_With_Emb_Layer_Scratch(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super(H_Critic_With_Emb_Layer_Scratch, self).__init__()
        self.hidden_size = args.hidden_size
        self.base = Critic_With_Emb_Layer_Scratch(args, args.node_num, args.defender_num, device,
                                                  dynamic_hidden_dim=self.hidden_size,
                                                  head_init_method=args.h_init)
        self.to(device)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, hgs, T, cent_obs, batch=False):
        values = self.base(hgs, T, cent_obs, batch=batch)
        return values