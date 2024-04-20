import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grasper_mappo.utils.util import get_gard_norm, huber_loss
from grasper_mappo.utils.valuenorm import ValueNorm

class RMAPPO:
    def __init__(self, mappo_args, args, policy, device=torch.device("cpu")):
        self.mappo_args = mappo_args
        self.device = device
        self.policy = policy
        self.clip_param = mappo_args.clip_param
        self.ppo_epoch = mappo_args.ppo_epoch
        self.value_loss_coef = mappo_args.value_loss_coef
        self.entropy_coef = mappo_args.entropy_coef
        self.act_sup_coef_min = args.act_sup_coef_min
        self.act_sup_coef_max = args.act_sup_coef_max
        self.act_sup_coef_decay = args.act_sup_coef_decay
        self.act_sup_coef = self.act_sup_coef_max
        self.max_grad_norm = mappo_args.max_grad_norm
        self.huber_delta = mappo_args.huber_delta
        self._use_max_grad_norm = mappo_args.use_max_grad_norm
        self._use_clipped_value_loss = mappo_args.use_clipped_value_loss
        self._use_huber_loss = mappo_args.use_huber_loss
        self._use_popart = mappo_args.use_popart
        self._use_valuenorm = mappo_args.use_valuenorm
        self._use_advnorm = mappo_args.use_advnorm
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self._step = int(args.checkpoint / (args.num_games * args.num_task * args.num_sample))

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = F.mse_loss(return_batch, value_pred_clipped)
            value_loss_original = F.mse_loss(return_batch, values)
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        if self._use_huber_loss:
            value_loss = value_loss.mean()
        return value_loss

    def act_sup_coef_linear_decay(self):
        if self._step > self.act_sup_coef_decay:
            self.act_sup_coef = self.act_sup_coef_min
        else:
            self.act_sup_coef = self.act_sup_coef_max - self.act_sup_coef_max * (self._step / float(self.act_sup_coef_decay))

    def ppo_update(self, sample, update_actor=True):
        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, demo_act_probs_batch, pooled_node_embs_batch = sample
        values, action_log_probs, dist_entropy, log_probs = self.policy.evaluate_actions(share_obs_batch, obs_batch, actions_batch, pooled_node_embs_batch, batch=True)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            if demo_act_probs_batch is not None:
                (policy_loss - dist_entropy * self.entropy_coef + self.act_sup_coef * F.kl_div(log_probs, demo_act_probs_batch, reduction="batchmean")).backward()
            else:
                (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, demo_act_probs_batch

    def train(self, buffer, update_actor=True, train_num_per_ite=1):
        trained = False
        train_info = {'value_loss': 0, 'policy_loss': 0, 'dist_entropy': 0, 'actor_grad_norm': 0, 'critic_grad_norm': 0, 'ratio': 0}
        total_transition_num = len(buffer.share_obs) * buffer.num_agent
        if total_transition_num > buffer.batch_size:
            trained = True
            if self._use_popart or self._use_valuenorm:
                advantages = np.array(buffer.returns) - self.value_normalizer.denormalize(np.array(buffer.value_preds))
            else:
                advantages = np.array(buffer.returns) - np.array(buffer.value_preds)
            if self._use_advnorm:
                advantages_copy = advantages.copy()
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            for _ in range(train_num_per_ite):
                sample = buffer.get_batch(advantages, self.device)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, demo_act_probs_batch = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
            if train_num_per_ite > 1:
                for k in train_info.keys():
                    train_info[k] /= train_num_per_ite
            self._step += 1
            if demo_act_probs_batch is not None and self.act_sup_coef_min != self.act_sup_coef_max:
                self.act_sup_coef_linear_decay()
        return train_info, trained

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

