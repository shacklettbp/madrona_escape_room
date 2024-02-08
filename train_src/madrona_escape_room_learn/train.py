import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
import os
from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict
from .profile import profile
from time import time
from pathlib import Path
import os

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .learning_state import LearningState
from .replay_buffer import NStepReplay

import datetime

@dataclass(frozen = True)
class MiniBatch:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


@dataclass
class PPOStats:
    loss : float = 0
    action_loss : float = 0
    value_loss : float = 0
    entropy_loss : float = 0
    returns_mean : float = 0
    returns_stddev : float = 0


@dataclass(frozen = True)
class UpdateResult:
    obs: torch.Tensor
    actions : torch.Tensor
    rewards : torch.Tensor
    returns : torch.Tensor
    dones: torch.Tensor
    values : torch.Tensor
    advantages : torch.Tensor
    bootstrap_values : torch.Tensor
    ppo_stats : PPOStats


def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:])[:, inds, ...]

def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4])

    return reshaped[:, :, inds, :] 

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPState):
    obs_slice = tuple(_mb_slice(obs, inds) for obs in rollouts.obs)

    # Print if in third room
    #third_room_count = (obs_slice[0][:,3] > 0.65).sum()
    #if third_room_count > 0:
    #    print("There are ", third_room_count, "agents in the third room")
    
    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(
        dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(
        dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(
        dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(
        dtype=amp.compute_dtype)

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states)

    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        advantages=advantages_slice,
        rnn_start_states=rnn_starts_slice,
    )

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPState,
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        if cfg.ppo.no_advantages:
            td_err = (cur_rewards + 
                cfg.gamma * next_valid * next_values) # Don't subtract off cur_values
        else:
            td_err = (cur_rewards + 
                cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values

def _compute_action_scores(cfg, amp, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var.clamp(min=1e-5)))

            return action_scores.to(dtype=amp.compute_dtype)

# Keep value loss the same as PPO, but make the policy loss be advantage-weighted regression
# Introduce buffer if needed
def _awr_update(cfg : TrainConfig,
                amp : AMPState,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
            action_scores = _compute_action_scores(cfg, amp, mb.advantages)

        '''
        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        action_obj = torch.min(surr1, surr2)
        '''
        action_obj = new_log_probs * torch.exp(action_scores * cfg.awr.beta_inverse) 

        returns = mb.advantages + mb.values

        if cfg.awr.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef

            new_values = torch.clamp(new_values, low, high)

        normalized_returns = value_normalizer(amp, returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj # Maximize the action objective function
            + cfg.ppo.value_loss_coef * value_loss
            - cfg.ppo.entropy_coef * entropies # Maximize entropy
        )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = value_loss.cpu().float().item(),
            entropy_loss = -(entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
        )

    return stats

def _ppo_update(cfg : TrainConfig,
                amp : AMPState,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
            action_scores = _compute_action_scores(cfg, amp, mb.advantages)

        #if mb.dones.sum() > 0: # VISHNU LOGGING
        #    print("We have a done!")

        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        action_obj = torch.min(surr1, surr2)

        returns = mb.advantages + mb.values

        if cfg.ppo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef

            new_values = torch.clamp(new_values, low, high)

        normalized_returns = value_normalizer(amp, returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj # Maximize the action objective function
            + cfg.ppo.value_loss_coef * value_loss
            - cfg.ppo.entropy_coef * entropies # Maximize entropy
        )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            #print("MAX")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.max(v.grad))
            #print("MIN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.min(v.grad))
            #print("MEAN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.mean(v.grad))

            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            #print("MAX")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.max(v.grad))
            #print("MIN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.min(v.grad))
            #print("MEAN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.mean(v.grad))
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = (cfg.ppo.value_loss_coef * value_loss.cpu().float().item()),
            entropy_loss = -(cfg.ppo.entropy_coef * entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
        )

    return stats

def _update_iter(cfg : TrainConfig,
                 amp : AMPState,
                 num_train_seqs : int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 advantages : torch.Tensor,
                 actor_critic : ActorCritic,
                 optimizer : torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler.LRScheduler,
                 value_normalizer : EMANormalizer,
                 replay_buffer: NStepReplay,
                 user_cb,
            ):
    with torch.no_grad():
        actor_critic.eval()
        value_normalizer.eval()
        # This is where the simulator loop happens that executes the TaskGraph.
        with profile('Collect Rollouts'):
            rollouts = rollout_mgr.collect(amp, sim, actor_critic, value_normalizer)
            #print("Testing: adding to buffer")
            #replay_buffer.add_to_buffer(rollouts)
            #print("Testing: load oldest thing in buffer")
            #rollouts = replay_buffer.get_last(rollouts)
            #print("Testing: load multiple from buffer")
            #rollouts = replay_buffer.get_multiple(rollouts)
            
            # Now modify the rewards in the rollouts by adding reward when closer to "exit"
            if type(user_cb).__name__ == "GoExplore":
                # Compute change in exit dist
                all_bins = user_cb.map_states_to_bins(rollouts.obs) # num_timesteps * num_worlds
                #if user_cb.max_progress < 1.01:
                reward_bonus_1 = user_cb.start_bin_steps[all_bins]
                #print(reward_bonus_1.sum(axis=0))
                #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] *= 0
                rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] += reward_bonus_1[...,None].repeat(1,1,2).view(reward_bonus_1.shape[0],-1,1) * user_cb.bin_reward_boost * 0.5
                max_bin_steps = 200
                if user_cb.bin_steps[user_cb.bin_steps < 200].size(dim=0) > 0:
                    max_bin_steps = user_cb.bin_steps[user_cb.bin_steps < 200].max()
                reward_bonus_2 = max_bin_steps - user_cb.bin_steps[all_bins]
                reward_bonus_2[reward_bonus_2 < 0] = 0
                #reward_bonus = (user_cb.bin_steps[all_bins[1:]] < user_cb.bin_steps[all_bins[:-1]]).float() - (user_cb.bin_steps[all_bins[1:]] > user_cb.bin_steps[all_bins[:-1]]).float()
                #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:-1] += reward_bonus[...,None].repeat(1,1,2).view(reward_bonus.shape[0],-1,1) * user_cb.bin_reward_boost
                #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] += reward_bonus_2[...,None].repeat(1,1,2).view(reward_bonus_2.shape[0],-1,1) * user_cb.bin_reward_boost

        # Dump the rollout
        '''
        curr_rand = torch.rand((1,))[0]
        if curr_rand < 0.05:
            # Dump the features
            now = datetime.datetime.now()
            dir_path = "/data/rl/madrona_3d_example/data_dump/" + cfg.run_name + "/"
            isExist = os.path.exists(dir_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(dir_path)
            torch.save(rollouts, dir_path + str(now) + ".pt")
        '''
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            _compute_advantages(cfg,
                                amp,
                                advantages,
                                rollouts)
    
    actor_critic.train()
    value_normalizer.train()

    with profile('PPO'):
        aggregate_stats = PPOStats()
        num_stats = 0

        for epoch in range(cfg.ppo.num_epochs):
            #for inds in torch.arange(num_train_seqs).chunk(
            for inds in torch.randperm(num_train_seqs).chunk(
                    cfg.ppo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = _gather_minibatch(rollouts, advantages, inds, amp)
                cur_stats = _ppo_update(cfg,
                                        amp,
                                        mb,
                                        actor_critic,
                                        optimizer,
                                        value_normalizer)

                with torch.no_grad():
                    num_stats += 1
                    aggregate_stats.loss += (cur_stats.loss - aggregate_stats.loss) / num_stats
                    aggregate_stats.action_loss += (
                        cur_stats.action_loss - aggregate_stats.action_loss) / num_stats
                    aggregate_stats.value_loss += (
                        cur_stats.value_loss - aggregate_stats.value_loss) / num_stats
                    aggregate_stats.entropy_loss += (
                        cur_stats.entropy_loss - aggregate_stats.entropy_loss) / num_stats
                    aggregate_stats.returns_mean += (
                        cur_stats.returns_mean - aggregate_stats.returns_mean) / num_stats
                    # FIXME
                    aggregate_stats.returns_stddev += (
                        cur_stats.returns_stddev - aggregate_stats.returns_stddev) / num_stats

    return UpdateResult(
        obs = rollouts.obs,
        actions = rollouts.actions.view(-1, *rollouts.actions.shape[2:]),
        rewards = rollouts.rewards.view(-1, *rollouts.rewards.shape[2:]),
        returns = rollouts.returns.view(-1, *rollouts.returns.shape[2:]),
        dones = rollouts.dones.view(-1, *rollouts.dones.shape[2:]),
        values = rollouts.values.view(-1, *rollouts.values.shape[2:]),
        advantages = advantages.view(-1, *advantages.shape[2:]),
        bootstrap_values = rollouts.bootstrap_values,
        ppo_stats = aggregate_stats,
    )

def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 learning_state : LearningState,
                 start_update_idx : int,
                 replay_buffer: NStepReplay):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert(num_train_seqs % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)
    second_room_ckpts = torch.zeros_like(sim.checkpoints) # Once this is full, rotate as FIFO
    total_second_room_ckpts = 0
    third_room_ckpts = torch.zeros_like(sim.checkpoints) # Once this is full, rotate as FIFO
    total_third_room_ckpts = 0
    checkpoint_buffer_size = sim.checkpoints.shape[0] # Check that this is the right size
    print("Checkpoint buffer shape", second_room_ckpts.shape)

    useCKPT = False
    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        # Restore second-room and third-room checkpoints if there are any, every 5 steps
        if False:
            if total_second_room_ckpts > 0:
                # Set the first 2000 worlds to randomly-selected second room checkpoints
                checkpoint_indices = torch.randint(0, total_second_room_ckpts, (1000,))
                print("Checkpoint indices", checkpoint_indices.shape, checkpoint_indices[:10])
                print(second_room_ckpts[0])
                #print("Before setting checkpoints", sim.checkpoints[:2000])
                sim.checkpoints[:1000] = second_room_ckpts[checkpoint_indices]
                #print("After setting checkpoints", sim.checkpoints[:2000])
            if total_third_room_ckpts > 0:
                # Set the next 2000 worlds to randomly-selected third room checkpoints
                checkpoint_indices = torch.randint(0, total_third_room_ckpts, (3000,))
                sim.checkpoints[1000:4000] = third_room_ckpts[checkpoint_indices]
            # After reset, step to collect observations for the next rollout.
            if total_second_room_ckpts > 0:
                sim.resets[:, 0] = 1
                sim.checkpoint_resets[:, 0] = 1
                sim.step()
                print("Just ran a reset")

            # Print steps_remaining
            sim.obs[5][:8000] = 40 + torch.randint(0, 160, size=(8000,1), dtype=torch.int, device=sim.obs[5].device)

        #print("Update", update_idx)
        #print("Steps remaining", sim.obs[5][:8000])
        #print("Shape", sim.obs[5].shape, sim.checkpoints.shape)

        if useCKPT and update_idx > 0:
            # Run the minisim here to set state and initialize observations,
            #sim.resets[:, 0] = 1 # No longer necessary, happens automatically on checkpoint_reset. 
            sim.checkpoint_resets[:, 0] = 1
            sim.checkpoints[:] = torch.cat((sim.checkpoints[1:], sim.checkpoints[0:1]), dim=0)

            # After reset, step to collect observations for the next rollout.
            sim.step()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                learning_state.amp,
                num_train_seqs,
                sim,
                rollout_mgr,
                advantages,
                learning_state.policy,
                learning_state.optimizer,
                learning_state.scheduler,
                learning_state.value_normalizer,
                replay_buffer,
                user_cb,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, learning_state)

        # Check if we're in the 2nd or 3rd rooms, and potentially add to the checkpoint buffers
        # Does this kill the diversity if it's not coming from the "uncontrolled" worlds? 
        # We may just need to add a lot more checkpoints to the buffer
        # We can also start introducing the binning function here
        if False:
            print("Tensor shape", sim.obs[0].shape)
            per_world_tensor = sim.obs[0].reshape(-1, 2, 9) # 9 features, 2 agents
            third_room_flag = (per_world_tensor[...,3] > 0.67)
            second_room_flag = (per_world_tensor[...,3] > 0.34) ^ third_room_flag
            third_room_flag = third_room_flag.sum(dim=1) > 0
            second_room_flag = second_room_flag.sum(dim=1) > 0
            second_room_flag[:4000] = 0 # Only add instances from fresh worlds
            third_room_flag[:4000] = 0 # Only add instances from fresh worlds
            num_second_room = second_room_flag.sum()
            num_third_room = third_room_flag.sum()
            print(num_second_room, "agents in second room")
            print(num_third_room, "agents in third room")
            if num_second_room > 0:
                print("Adding", sim.checkpoints[second_room_flag])
                second_room_ckpts = torch.cat((sim.checkpoints[second_room_flag], second_room_ckpts[:-num_second_room]), dim=0)
                if total_second_room_ckpts < checkpoint_buffer_size:
                    total_second_room_ckpts += min(checkpoint_buffer_size - total_second_room_ckpts, num_second_room)
            if num_third_room > 0:
                third_room_ckpts = torch.cat((sim.checkpoints[third_room_flag], third_room_ckpts[:-num_third_room]), dim=0)
                if total_third_room_ckpts < checkpoint_buffer_size:
                    total_third_room_ckpts += min(checkpoint_buffer_size - total_third_room_ckpts, num_third_room)

def train(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    amp = AMPState(dev, cfg.mixed_precision)

    value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                     disable=not cfg.normalize_values)
    value_normalizer = value_normalizer.to(dev)

    learning_state = LearningState(
        policy = actor_critic,
        optimizer = optimizer,
        scheduler = None,
        value_normalizer = value_normalizer,
        amp = amp,
    )

    # Restore a previous policy, nothing to do with the state of the world.
    if restore_ckpt != None:
        start_update_idx = learning_state.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    buffer_size = 15
    replay_buffer = NStepReplay(rollout_mgr, buffer_size, 'cuda')

    _update_loop(
        update_iter_fn=_update_iter,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        learning_state=learning_state,
        start_update_idx=start_update_idx,
        replay_buffer=replay_buffer,
    )

    return actor_critic
