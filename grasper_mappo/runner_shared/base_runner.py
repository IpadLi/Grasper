import torch
import numpy as np
import dgl

from grasper_mappo.utils.shared_buffer import SharedReplayBuffer, SharedReplayBufferFT, SharedReplayBufferEnd2End
from grasper_mappo.algorithms.algorithm.r_hmappo import RHMAPPO as TrainAlgo
from grasper_mappo.algorithms.algorithm.rHMAPPOPolicy import RHMAPPOPolicy as Policy
from grasper_mappo.algorithms.algorithm.rHMAPPOPolicy import RHMAPPOPolicyScratch as PolicyScratch
from grasper_mappo.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgoFT
from grasper_mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as PolicyFT

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, env, mappo_args, args):
        self.mappo_args = mappo_args
        self.args = args
        self.env = env
        self.device = args.device
        self.num_agents = self.env.agent_num
        self.num_defender = self.env.defender_num
        self.env.initialize_attacker_strategy()
        self.defender_index = np.eye(self.num_defender)[np.arange(self.num_defender)].astype(np.int32)

        # used during pre-training
        share_obs_size = self.env.share_obs_dim  # evder loc + all locs + time step
        obs_size = share_obs_size + 1
        if args.use_end_to_end:
            self.policy = PolicyScratch(self.mappo_args, self.args, obs_size, share_obs_size, self.env.action_dim, device=self.device)
            self.buffer = SharedReplayBufferEnd2End(self.mappo_args, self.args, self.env.share_obs_dim, self.env.obs_dim, self.env.action_dim, self.num_defender)
        else:
            hyper_input_dim = args.gnn_output_dim + args.max_time_horizon_for_state_emb
            self.policy = Policy(self.mappo_args, self.args, obs_size, share_obs_size, hyper_input_dim, self.env.action_dim, device=self.device)
            self.buffer = SharedReplayBuffer(self.mappo_args, self.args, self.env.share_obs_dim, self.env.obs_dim, self.env.action_dim, self.num_defender)

        self.trainer = TrainAlgo(self.mappo_args, self.args, self.policy, device=self.device)
        if args.load_pretrain_model:
            print("Load hyper model from checkpoint {}: {}/{}".format(self.args.pretrain_model_iteration, args.actor_model, args.critic_model))
            self.load_checkpoint(args.actor_model, args.critic_model)

        # used during fine-tunning
        self.policy_ft = PolicyFT(self.mappo_args, self.args, obs_size, share_obs_size, self.env.action_dim, device=self.device)
        self.trainer_ft = TrainAlgoFT(self.mappo_args, self.args, self.policy_ft, device=self.device)
        self.buffer_ft = SharedReplayBufferFT(self.mappo_args, self.args, share_obs_size, obs_size, self.env.action_dim, self.num_defender)

    def load_checkpoint(self, actor_checkpoint, critic_checkpoint):
        self.trainer.policy.actor.load_state_dict(torch.load(actor_checkpoint))
        self.trainer.policy.critic.load_state_dict(torch.load(critic_checkpoint))

    def reset(self, game):
        self.env.reset_by_game(game)
        self.env.initialize_attacker_strategy()

    def save(self, file_name):
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), file_name + "_actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), file_name + "_critic.pt")

    #=============== used by pretrained GNN training methods ==================
    def collect(self, pooled_node_embs, Ts, shared_obs, obs):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self, pooled_node_embs, Ts, shared_obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
        Ts = torch.FloatTensor(Ts).to(self.device)
        pooled_node_embs = torch.FloatTensor(pooled_node_embs).to(self.device)
        next_values = self.trainer.policy.get_values(pooled_node_embs, Ts, shared_obs, batch=True)
        next_values = _t2n(next_values)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        self.trainer.prep_training()
        train_infos, trained = self.trainer.train(self.buffer)
        if trained:
            self.buffer.after_update()
        return train_infos

    #==================== used by end to end GNN training method ============
    def collect_e2e(self, hgs, Ts, shared_obs, obs):
        raise NotImplementedError

    def insert_e2e(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute_e2e(self, hgs, Ts, shared_obs):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
        Ts = torch.FloatTensor(Ts).to(self.device)
        hgs_batch = dgl.batch(hgs)
        hgs_batch = hgs_batch.to(self.device)
        next_values = self.trainer.policy.get_values(hgs_batch, Ts, shared_obs, batch=True)
        next_values = _t2n(next_values)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    #======================= used by fine-tunning method ==============================
    def collect_ft(self, shared_obs, obs):
        raise NotImplementedError

    def insert_ft(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute_ft(self, shared_obs, pooled_node_emb):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
        next_values = self.trainer_ft.policy.get_values(shared_obs, pooled_node_emb, batch=True)
        next_values = _t2n(next_values)
        self.buffer_ft.compute_returns(next_values, self.trainer_ft.value_normalizer)

    def train_ft(self, train_num_per_ite=1):
        self.trainer_ft.prep_training()
        train_infos, trained = self.trainer_ft.train(self.buffer_ft, train_num_per_ite=train_num_per_ite)
        if trained:
            self.buffer_ft.after_update()
        return train_infos