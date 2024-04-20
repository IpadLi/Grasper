import torch

from mappo.utils.shared_buffer import SharedReplayBuffer
from mappo.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
from mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, env, mappo_args, args, train_mode='pretrain'):
        self.mappo_args = mappo_args
        self.args = args
        self.train_mode = train_mode
        self.env = env
        self.device = args.device
        self.num_agents = self.env.agent_num
        self.num_defender = self.env.defender_num
        self.env.initialize_attacker_strategy()

        share_obs_size = self.env.share_obs_dim # evder loc + all locs + time step
        obs_size = share_obs_size + 1
        self.policy = Policy(self.mappo_args, self.args, obs_size, share_obs_size, self.env.action_dim, device=self.device)
        self.trainer = TrainAlgo(self.mappo_args, self.args, self.policy, device=self.device)
        self.buffer = SharedReplayBuffer(self.mappo_args, self.args, share_obs_size, obs_size, self.env.action_dim, self.num_defender, train_mode)
        if args.load_pretrain_model:
            print("Load mappo model from checkpoint {}: {}/{}".format(self.args.pretrain_model_iteration, args.actor_model, args.critic_model))
            self.load_checkpoint(args.actor_model, args.critic_model)

    def load_checkpoint(self, actor_checkpoint, critic_checkpoint):
        self.trainer.policy.actor.load_state_dict(torch.load(actor_checkpoint))
        self.trainer.policy.critic.load_state_dict(torch.load(critic_checkpoint))

    def reset(self, game):
        self.env.reset_by_game(game)
        self.env.initialize_attacker_strategy()

    def collect(self, shared_obs, obs, pooled_node_emb, Ts):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, shared_obs, pooled_node_emb, Ts):
        if self.args.use_emb_layer:
            shared_obs = torch.LongTensor(shared_obs).to(self.device)
        else:
            shared_obs = torch.FloatTensor(shared_obs).to(self.device)
        next_values = self.trainer.policy.get_values(shared_obs, pooled_node_emb, Ts, batch=True)
        next_values = _t2n(next_values)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self, train_num_per_ite=1):
        train_epoch = self.trainer.ppo_epoch if self.train_mode == 'pretrain' else train_num_per_ite
        self.trainer.prep_training()
        train_infos, trained = self.trainer.train(self.buffer, train_num_per_ite=train_epoch)
        if trained:
            self.buffer.after_update()
        return train_infos

    def save(self, file_name):
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), file_name + "_actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), file_name + "_critic.pt")