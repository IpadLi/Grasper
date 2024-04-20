import numpy as np
import torch
import dgl

from grasper_mappo.runner_shared.base_runner import Runner
from utils.utils import get_demonstration

def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, env, mappo_args, args):
        super(EnvRunner, self).__init__(env, mappo_args, args)

    def run_one_episode(self, pooled_node_embs, Ts, use_act_supervisor=True):
        episode_reward, episode_length = 0, 0
        shared_obs, obs, time_step = self.env.reset()  # shape = (n_agent, obs_size)
        if use_act_supervisor:
            demonstration_distribs = get_demonstration(obs, self.env.game, self.env.attacker_path[-1])
        else:
            demonstration_distribs = None
        obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)  # include own id
        while not time_step.last():
            values, actions, action_log_probs, actions_env = self.collect(pooled_node_embs, Ts, shared_obs, obs)
            shared_obs_next, obs_next, rewards, dones, time_step = self.env.step(actions_env)
            data = pooled_node_embs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs
            self.insert(data)
            episode_reward += rewards[0,0]
            if use_act_supervisor:
                demonstration_distribs_next = get_demonstration(obs_next, self.env.game, self.env.attacker_path[-1])
            else:
                demonstration_distribs_next = None
            obs_next = np.concatenate((shared_obs_next, np.expand_dims(obs_next[:, -1], 1)), axis=1)  # include own id
            shared_obs, obs = shared_obs_next.copy(), obs_next.copy()
            if demonstration_distribs_next is not None:
                demonstration_distribs = demonstration_distribs_next.copy()
            episode_length += 1
        self.buffer.episode_length.append(episode_length)
        self.compute(pooled_node_embs, Ts, shared_obs)
        return episode_reward

    @torch.no_grad()
    def collect(self, pooled_node_embs, Ts, shared_obs, obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
            obs = torch.LongTensor(obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
        Ts = torch.FloatTensor(Ts).to(self.device)
        pooled_node_embs = torch.FloatTensor(pooled_node_embs).to(self.device)
        value, action, action_log_prob = self.trainer.policy.get_actions(pooled_node_embs, Ts, shared_obs, obs, batch=True)
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        actions_env = np.reshape(actions, self.num_defender)
        return values, actions, action_log_probs, actions_env

    def insert(self, data):
        pooled_node_embs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs = data
        masks = np.ones((self.num_defender, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        self.buffer.insert(pooled_node_embs, Ts, shared_obs, obs, actions, action_log_probs, values, rewards, masks, demonstration_distribs)

    def run_one_episode_e2e(self, hgs, Ts, use_act_supervisor=True):
        episode_reward, episode_length = 0, 0
        shared_obs, obs, time_step = self.env.reset()  # shape = (n_agent, obs_size)
        if use_act_supervisor:
            demonstration_distribs = get_demonstration(obs, self.env.game, self.env.attacker_path[-1])
        else:
            demonstration_distribs = None
        obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)  # include own id
        while not time_step.last():
            values, actions, action_log_probs, actions_env = self.collect_e2e(hgs, Ts, shared_obs, obs)
            shared_obs_next, obs_next, rewards, dones, time_step = self.env.step(actions_env)
            data = hgs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs
            self.insert_e2e(data)
            episode_reward += rewards[0,0]
            if use_act_supervisor:
                demonstration_distribs_next = get_demonstration(obs_next, self.env.game, self.env.attacker_path[-1])
            else:
                demonstration_distribs_next = None
            obs_next = np.concatenate((shared_obs_next, np.expand_dims(obs_next[:, -1], 1)), axis=1)  # include own id
            shared_obs, obs = shared_obs_next.copy(), obs_next.copy()
            if demonstration_distribs_next is not None:
                demonstration_distribs = demonstration_distribs_next.copy()
            episode_length += 1
        self.buffer.episode_length.append(episode_length)
        self.compute_e2e(hgs, Ts, shared_obs)
        return episode_reward

    @torch.no_grad()
    def collect_e2e(self, graphs, Ts, shared_obs, obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
            obs = torch.LongTensor(obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
        Ts = torch.FloatTensor(Ts).to(self.device)
        hgs_batch = dgl.batch(graphs)
        hgs_batch = hgs_batch.to(self.device)
        value, action, action_log_prob = self.trainer.policy.get_actions(hgs_batch, Ts, shared_obs, obs, batch=True)
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        actions_env = np.reshape(actions, self.num_defender)
        return values, actions, action_log_probs, actions_env

    def insert_e2e(self, data):
        hgs, Ts, shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs = data
        masks = np.ones((self.num_defender, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        self.buffer.insert(hgs, Ts, shared_obs, obs, actions, action_log_probs, values, rewards, masks, demonstration_distribs)

    def run_one_episode_ft(self, pooled_node_emb, use_act_supervisor=True):
        episode_reward, episode_length = 0, 0
        shared_obs, obs, time_step = self.env.reset()  # shape = (n_agent, obs_size)
        if use_act_supervisor:
            demonstration_distribs = get_demonstration(obs, self.env.game, self.env.attacker_path[-1])
        else:
            demonstration_distribs = None
        obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)  # include own id
        if not self.args.use_augmentation:
            pooled_node_emb = None
        while not time_step.last():
            values, actions, action_log_probs, actions_env = self.collect_ft(shared_obs, obs, pooled_node_emb)
            shared_obs_next, obs_next, rewards, dones, time_step = self.env.step(actions_env)
            data = shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs, pooled_node_emb
            self.insert_ft(data)
            episode_reward += rewards[0,0]
            if use_act_supervisor:
                demonstration_distribs_next = get_demonstration(obs_next, self.env.game, self.env.attacker_path[-1])
            else:
                demonstration_distribs_next = None
            obs_next = np.concatenate((shared_obs_next, np.expand_dims(obs_next[:, -1], 1)), axis=1)  # include own id
            shared_obs, obs = shared_obs_next.copy(), obs_next.copy()
            if demonstration_distribs_next is not None:
                demonstration_distribs = demonstration_distribs_next.copy()
            episode_length += 1
        self.buffer_ft.episode_length.append(episode_length)
        self.compute_ft(shared_obs, pooled_node_emb)
        return episode_reward

    @torch.no_grad()
    def collect_ft(self, shared_obs, obs, pooled_node_emb):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
            obs = torch.LongTensor(obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
            obs = torch.FloatTensor(obs).to(self.device)
        if self.args.use_augmentation:
            pooled_node_emb = torch.FloatTensor(pooled_node_emb).to(self.device)
        value, action, action_log_prob = self.trainer_ft.policy.get_actions(shared_obs, obs, pooled_node_emb, batch=True)
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        actions_env = np.reshape(actions, self.num_defender)
        return values, actions, action_log_probs, actions_env

    def insert_ft(self, data):
        shared_obs, obs, rewards, dones, values, actions, action_log_probs, demonstration_distribs, pooled_node_emb = data
        masks = np.ones((self.num_defender, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        self.buffer_ft.insert(shared_obs, obs, actions, action_log_probs, values, rewards, masks, demonstration_distribs, pooled_node_emb)