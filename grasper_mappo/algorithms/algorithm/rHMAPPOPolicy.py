import torch

from grasper_mappo.algorithms.algorithm.h_actor_critic import (
    H_Actor,
    H_Critic,
    H_Actor_With_Emb_Layer,
    H_Critic_With_Emb_Layer,
    H_Actor_With_Emb_Layer_Scratch,
    H_Critic_With_Emb_Layer_Scratch
)


class RHMAPPOPolicy:
    def __init__(self, mappo_args, args, obs_space, cent_obs_space, hyper_input_dim, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = mappo_args.lr
        self.critic_lr = mappo_args.critic_lr
        self.opti_eps = mappo_args.opti_eps
        self.weight_decay = mappo_args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.hyper_input_dim = hyper_input_dim
        self.act_space = act_space

        if args.use_emb_layer:
            self.actor = H_Actor_With_Emb_Layer(args, hyper_input_dim, self.act_space, self.device)
            self.critic = H_Critic_With_Emb_Layer(args, hyper_input_dim, self.device)
        else:
            self.actor = H_Actor(args, self.obs_space, self.hyper_input_dim, self.act_space, self.device)
            self.critic = H_Critic(args, self.share_obs_space, self.hyper_input_dim, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, pooled_node_emb, T, cent_obs, obs, batch=False):
        actions, action_log_probs = self.actor(pooled_node_emb, T, obs, batch=batch)
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, pooled_node_emb, T, cent_obs, batch=False):
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values

    def evaluate_actions(self, pooled_node_emb, T, cent_obs, obs, action, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(pooled_node_emb, T, obs, action, batch=batch)
        values = self.critic(pooled_node_emb, T, cent_obs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, pooled_node_emb, T, obs, batch=False):
        actions, _ = self.actor(pooled_node_emb, T, obs, batch=batch)
        return actions


class RHMAPPOPolicyScratch:
    def __init__(self, mappo_args, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = mappo_args.lr
        self.critic_lr = mappo_args.critic_lr
        self.opti_eps = mappo_args.opti_eps
        self.weight_decay = mappo_args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if args.use_emb_layer:
            self.actor = H_Actor_With_Emb_Layer_Scratch(args, self.act_space, self.device)
            self.critic = H_Critic_With_Emb_Layer_Scratch(args, self.device)
        else:
            raise ValueError("Must use observation embedding layer.")
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, hgs, T, cent_obs, obs, batch=False):
        actions, action_log_probs = self.actor(hgs, T, obs, batch=batch)
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, hgs, T, cent_obs, batch=False):
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values

    def evaluate_actions(self, hgs, T, cent_obs, obs, action, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(hgs, T, obs, action, batch=batch)
        values = self.critic(hgs, T, cent_obs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, hgs, T, obs, batch=False):
        actions, _ = self.actor(hgs, T, obs, batch=batch)
        return actions