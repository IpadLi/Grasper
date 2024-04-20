import torch
import numpy as np
import dgl

class SharedReplayBuffer(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.pooled_node_emb_shape = args.gnn_output_dim
        self.T_shape = args.max_time_horizon_for_state_emb
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.pooled_node_embs = []
        self.Ts = []
        self.value_preds = []
        self.returns = []
        self.actions = []
        self.demo_act_probs = []
        self.action_log_probs = []
        self.rewards = []
        self.masks = []
        self.episode_length = []

        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []


    def insert(self, pooled_node_embs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.pooled_node_embs.append(pooled_node_embs.copy())
        self.Ts.append(Ts.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.value_preds_one_episode.append(value_preds.copy())
        self.rewards_one_episode.append(rewards.copy())
        self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        self.masks_one_episode.append(masks.copy())
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())

    def after_update(self):
        del self.pooled_node_embs[:]  # clear experience
        del self.Ts[:]
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards_one_episode))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards_one_episode[step] + self.gamma * value_normalizer.denormalize(self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - value_normalizer.denormalize(self.value_preds_one_episode[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                else:
                    delta = self.rewards_one_episode[step] + self.gamma * (self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - self.value_preds_one_episode[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                    self.returns_one_episode[step] = gae + self.value_preds_one_episode[step]
        else:
            for step in reversed(range(len(self.rewards_one_episode))):
                self.returns_one_episode[step] = (self.returns_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                                                 * self.gamma * self.masks_one_episode[step] + self.rewards_one_episode[step]
        self.returns.extend(self.returns_one_episode)
        del self.value_preds_one_episode[:]
        del self.rewards_one_episode[:]
        del self.returns_one_episode[:]
        del self.masks_one_episode[:]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.pooled_node_embs) * self.num_agent
        batch_size = min(total_transition_num, self.batch_size)
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:batch_size]
        pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_embs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return pooled_node_embs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch


class SharedReplayBufferEnd2End(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.T_shape = args.max_time_horizon_for_state_emb
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.hgs = []
        self.Ts = []
        self.value_preds = []
        self.returns = []
        self.actions = []
        self.demo_act_probs = []
        self.action_log_probs = []
        self.rewards = []
        self.masks = []
        self.episode_length = []
        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

    def insert(self, hgs, Ts, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.hgs.append(hgs)
        self.Ts.append(Ts.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.value_preds_one_episode.append(value_preds.copy())
        self.rewards_one_episode.append(rewards.copy())
        self.returns_one_episode.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        self.masks_one_episode.append(masks.copy())
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())

    def after_update(self):
        del self.hgs[:]  # clear experience
        del self.Ts[:]
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards_one_episode))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards_one_episode[step] + self.gamma * value_normalizer.denormalize(self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - value_normalizer.denormalize(self.value_preds_one_episode[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                else:
                    delta = self.rewards_one_episode[step] + self.gamma * (self.value_preds_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                            * self.masks_one_episode[step] - self.value_preds_one_episode[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks_one_episode[step] * gae
                    self.returns_one_episode[step] = gae + self.value_preds_one_episode[step]
        else:
            for step in reversed(range(len(self.rewards_one_episode))):
                self.returns_one_episode[step] = (self.returns_one_episode[step + 1] if step < len(self.rewards_one_episode) - 1 else next_value) \
                                                 * self.gamma * self.masks_one_episode[step] + self.rewards_one_episode[step]
        self.returns.extend(self.returns_one_episode)
        del self.value_preds_one_episode[:]
        del self.rewards_one_episode[:]
        del self.returns_one_episode[:]
        del self.masks_one_episode[:]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.share_obs) * self.num_agent
        batch_size = min(total_transition_num, self.batch_size)
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:batch_size]
        graphs = [self.hgs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices]
        hgs_batch = dgl.batch(graphs)
        hgs_batch = hgs_batch.to(device)
        Ts_batch = torch.FloatTensor(np.array([self.Ts[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        return hgs_batch, Ts_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch


class SharedReplayBufferFT(object):
    def __init__(self, mappo_args, args, share_obs_shape, obs_shape, act_space, num_agent):
        self.args = args
        self.batch_size = 32
        self.gamma = mappo_args.gamma
        self.gae_lambda = mappo_args.gae_lambda
        self._use_gae = mappo_args.use_gae
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self.share_obs_shape = share_obs_shape
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.num_agent = num_agent

        self.share_obs = []
        self.obs = []
        self.pooled_node_emb = []
        self.value_preds = []
        self.returns = []
        self.actions = []
        self.demo_act_probs = []
        self.action_log_probs = []
        self.rewards = []
        self.masks = []
        self.episode_length = []
        self.value_preds_one_episode = []
        self.rewards_one_episode = []
        self.returns_one_episode = []
        self.masks_one_episode = []

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks, demo_act_probs=None, pooled_node_emb=None):
        self.share_obs.append(share_obs.copy())
        self.obs.append(obs.copy())
        self.value_preds.append(value_preds.copy())
        self.actions.append(actions.copy())
        self.action_log_probs.append(action_log_probs.copy())
        self.rewards.append(rewards.copy())
        self.masks.append(masks.copy())
        self.returns.append(np.zeros((obs.shape[0], 1), dtype=np.float32))
        if demo_act_probs is not None:
            self.demo_act_probs.append(demo_act_probs.copy())
        if pooled_node_emb is not None:
            self.pooled_node_emb.append(pooled_node_emb.copy())

    def after_update(self):
        del self.share_obs[:]
        del self.obs[:]
        del self.value_preds[:]
        del self.returns[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        if len(self.demo_act_probs) > 0:
            del self.demo_act_probs[:]
        if len(self.pooled_node_emb) > 0:
            del self.pooled_node_emb[:]
        del self.episode_length[:]

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            gae = 0
            for step in reversed(range(len(self.rewards))):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1] if step < len(self.rewards) - 1 else next_value) \
                            * self.masks[step] - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                else:
                    delta = self.rewards[step] + self.gamma * (self.value_preds[step + 1] if step < len(self.rewards) - 1 else next_value) \
                            * self.masks[step] - self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            for step in reversed(range(len(self.rewards))):
                self.returns[step] = (self.returns[step + 1] if step < len(self.rewards) - 1 else next_value) * self.gamma * self.masks[step] + self.rewards[step]

    def get_batch(self, advantages, device):
        total_transition_num = len(self.share_obs) * self.num_agent
        rand = torch.randperm(total_transition_num).numpy()
        indices = rand[:self.batch_size]
        if self.args.use_emb_layer:
            share_obs_batch = torch.LongTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.LongTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            share_obs_batch = torch.FloatTensor(np.array([self.share_obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
            obs_batch = torch.FloatTensor(np.array([self.obs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        actions_batch = torch.FloatTensor(np.array([self.actions[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        value_preds_batch = torch.FloatTensor(np.array([self.value_preds[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        return_batch = torch.FloatTensor(np.array([self.returns[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        masks_batch = torch.FloatTensor(np.array([self.masks[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        old_action_log_probs_batch = torch.FloatTensor(np.array([self.action_log_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = torch.FloatTensor(np.array([advantages[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        if len(self.demo_act_probs) > 0:
            demo_act_probs_batch = torch.FloatTensor(np.array([self.demo_act_probs[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            demo_act_probs_batch = None
        if len(self.pooled_node_emb) > 0:
            pooled_node_embs_batch = torch.FloatTensor(np.array([self.pooled_node_emb[int(ind // self.num_agent) - 1][int(ind % self.num_agent) - 1] for ind in indices])).to(device)
        else:
            pooled_node_embs_batch = None
        return share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch, pooled_node_embs_batch