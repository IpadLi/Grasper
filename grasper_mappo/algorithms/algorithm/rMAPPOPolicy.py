import torch

from grasper_mappo.algorithms.algorithm.r_actor_critic import R_Actor, R_Critic, R_Actor_With_Emb_Layer, R_Critic_With_Emb_Layer


class RMAPPOPolicy:
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
            self.actor = R_Actor_With_Emb_Layer(args, self.act_space, args.node_num, args.defender_num, self.device)
            self.critic = R_Critic_With_Emb_Layer(args, args.node_num, args.defender_num, self.device)
        else:
            self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
            self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, cent_obs, obs, pooled_node_embs, batch=False):
        actions, action_log_probs = self.actor(obs, pooled_node_embs, batch=batch)
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values, actions, action_log_probs

    def get_values(self, cent_obs, pooled_node_embs, batch=False):
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values

    def evaluate_actions(self, cent_obs, obs, action, pooled_node_embs, batch=False):
        action_log_probs, dist_entropy, action_probs = self.actor.evaluate_actions(obs, action, pooled_node_embs, batch=batch)
        values = self.critic(cent_obs, pooled_node_embs, batch=batch)
        return values, action_log_probs, dist_entropy, action_probs

    def act(self, obs, pooled_node_embs, batch=False):
        actions, _ = self.actor(obs, pooled_node_embs, batch=batch)
        return actions
