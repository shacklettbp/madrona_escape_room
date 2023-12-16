import madrona_escape_room
from madrona_escape_room import SimFlags, RewardMode

from madrona_escape_room_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface,
)

from policy import make_policy, setup_obs

import torch
import wandb
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
# General args
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--profile-report', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

# World gen args
arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--start-in-discovered-rooms', action='store_true')
arg_parser.add_argument('--reward-mode', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--gpu-sim', action='store_true')

# Learning args
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)
arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')
arg_parser.add_argument('--no-value-norm', action='store_true')
arg_parser.add_argument('--no-advantage-norm', action='store_true')
arg_parser.add_argument('--no-advantages', action='store_true')
arg_parser.add_argument('--value-normalizer-decay', type=float, default=0.999)
arg_parser.add_argument('--restore', type=int)
arg_parser.add_argument('--use-complex-level', action='store_true')


# Architecture args
arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')

# Go-Explore args
arg_parser.add_argument('--binning', type=str, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--num-bins', type=int, required=True)
arg_parser.add_argument('--num-checkpoints', type=int, default=1)
arg_parser.add_argument('--new-frac', type=float, default=0.5)

# Binning diagnostic args
arg_parser.add_argument('--bin-diagnostic', action='store_true')
arg_parser.add_argument('--seeds-per-checkpoint', type=int, default=16)

args = arg_parser.parse_args()

normalize_values = not args.no_value_norm
normalize_advantages = not args.no_advantage_norm

sim_flags = SimFlags.Default
print(sim_flags)
if args.use_fixed_world:
    sim_flags |= SimFlags.UseFixedWorld
if args.start_in_discovered_rooms:
    sim_flags |= SimFlags.StartInDiscoveredRooms
if args.use_complex_level:
    sim_flags |= SimFlags.UseComplexLevel
print(sim_flags)

reward_mode = getattr(RewardMode, args.reward_mode)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir = Path(args.ckpt_dir)

ckpt_dir.mkdir(exist_ok=True, parents=True)

from torch.distributions.categorical import Categorical

class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype=dtype

class DiscreteActionDistributions:
    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        self.dists = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            self.dists.append(Categorical(logits = logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                validate_args=False))
            cur_bucket_offset += num_buckets

    def best(self, out):
        actions = [dist.probs.argmax(dim=-1) for dist in self.dists]
        torch.stack(actions, dim=1, out=out)

    def sample(self):
        actions = [dist.sample() for dist in self.dists]
        return torch.stack(actions, dim=1)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]

class GoExplore:
    def __init__(self, num_worlds, device):
        self.worlds = madrona_escape_room.SimManager(
            exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
            gpu_id = args.gpu_id,
            num_worlds = args.num_worlds,
            auto_reset = True,
            sim_flags = sim_flags,
            reward_mode = reward_mode,
        )
        self.num_worlds = num_worlds
        self.num_agents = 2
        self.curr_returns = torch.zeros(num_worlds, device = device) # For tracking cumulative return of each state/bin
        #print("Curr returns shape", self.curr_returns.shape)
        self.binning = args.binning
        self.num_bins = args.num_bins # We can change this later
        self.num_checkpoints = args.num_checkpoints
        self.device = device
        self.checkpoint_score = torch.zeros(self.num_bins, self.num_checkpoints, device=device)
        self.bin_count = torch.zeros(self.num_bins, device=device).int()
        self.max_return = 0
        self.max_progress = 0

        self.obs, num_obs_features = setup_obs(self.worlds)
        self.policy = make_policy(num_obs_features, args.num_channels, args.separate_value)
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.checkpoints = self.worlds.checkpoint_tensor().to_torch()
        self.checkpoint_resets = self.worlds.checkpoint_reset_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()
        self.bin_checkpoints = torch.zeros((self.num_bins, self.num_checkpoints, self.checkpoints.shape[-1]), device=device, dtype=torch.uint8)
        self.bin_steps = torch.zeros((self.num_bins,), device=device).int() + 200
        self.world_steps = torch.zeros(self.num_worlds, device=device).int() + 200

        self.actions_num_buckets = [4, 8, 5, 2]
        self.action_space = Box(-float('inf'),float('inf'),(sum(self.actions_num_buckets),))

        # Callback
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = False

    # Corrected approach to get the first element of each group without using a for loop
    def get_first_elements_unsorted_groups(self, states, groups):
        # Sort groups and states based on groups
        sorted_groups, indices = groups.sort()
        sorted_states = states[indices]

        # Find the unique groups and the first occurrence of each group
        unique_groups, first_occurrences = torch.unique(sorted_groups, return_inverse=True)
        # Mask to identify first occurrences in the sorted array
        first_occurrence_mask = torch.zeros_like(sorted_groups, dtype=torch.bool).scatter_(0, first_occurrences, 1)

        return unique_groups, sorted_states[first_occurrence_mask]

    def generate_random_actions(self):
        action_dist = DiscreteActionDistributions(self.actions_num_buckets, logits=torch.ones(self.num_worlds*2, sum(self.actions_num_buckets), device=self.device))
        return action_dist.sample()

    # Step 1: Select state from archive
    # Uses: self.archive
    # Output: states
    def select_state(self):
        #print("About to select state")
        # First select from visited bins with go-explore weighting function
        valid_bins = torch.nonzero(self.bin_count > 0).flatten()
        weights = 1./(torch.sqrt(self.bin_count[valid_bins]) + 1)
        # Sample bins
        desired_samples = int(self.num_worlds*args.new_frac)
        sampled_bins = valid_bins[torch.multinomial(weights, num_samples=desired_samples, replacement=True).type(torch.int)]
        # Sample states from bins: either sample first occurrence in each bin (what's in the paper), or something better...
        # Need the last checkpoint for each bin
        chosen_checkpoint = (self.bin_count[sampled_bins] - 1) % self.num_checkpoints
        #chosen_checkpoint = torch.randint(0, self.num_checkpoints, size=(desired_samples,), device=self.device, dtype=torch.int)
        #chosen_checkpoint[chosen_checkpoint >= self.bin_count[sampled_bins]] = self.bin_count[sampled_bins][chosen_checkpoint >= self.bin_count[sampled_bins]] - 1
        #print("Chosen checkpoints", chosen_checkpoint, chosen_checkpoint.shape)
        #print("Bin count", self.bin_count[sampled_bins], self.bin_count[sampled_bins].shape)
        self.curr_returns[:desired_samples] = self.checkpoint_score[[sampled_bins, chosen_checkpoint]]
        print("Checkpoints", self.bin_checkpoints[[sampled_bins, chosen_checkpoint]])
        return self.bin_checkpoints[[sampled_bins, chosen_checkpoint]]

    # Step 2: Go to state
    # Input: states, worlds
    # Logic: For each state, set world to state
    # Output: None
    def go_to_state(self, states):
        # Run checkpoint-restoration here
        print("Before go-to-state")
        print(self.obs[5][:])
        desired_samples = int(self.num_worlds*args.new_frac) # Only set state for some worlds
        self.checkpoint_resets[:, 0] = 1
        self.checkpoints[:desired_samples] = states
        self.worlds.step()
        #self.obs[5][:desired_samples, 0] = 40*torch.randint(1, 6, size=(desired_samples,), dtype=torch.int, device=dev) # Maybe set this to random 40 to 200? # 200
        print("After go-to-state")
        print(self.obs[5][:])
        return None

    # Step 3: Explore from state
    def explore_from_state(self):
        for i in range(self.num_exploration_steps):
            # Select actions either at random or by sampling from a trained policy
            self.actions[:] = self.generate_random_actions()
            self.worlds.step()
            # Map states to bins
            new_bins = self.map_states_to_bins(self.obs)
            # Update archive
            #print(self.curr_returns.shape)
            #print(self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).shape)
            self.curr_returns += self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1)
            self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
            #print(self.dones.shape)
            #print(self.dones)
            self.curr_returns *= (1 - 0.5*self.dones.view(self.num_worlds,self.num_agents).sum(dim=1)) # Account for dones, this is working!
            #print("Max return", torch.max(self.curr_returns), self.worlds.obs[torch.argmax(self.curr_returns)])
            self.update_archive(new_bins, self.curr_returns)
        return None

    def apply_binning_function(self, states):
        if self.binning == "none":
            return states
        elif self.binning == "random":
            return torch.randint(0, self.num_bins, size=(self.num_worlds,), device=self.device)
        elif self.binning == "y_pos":
            # Bin according to the y position of each agent
            # Determine granularity from num_bins
            granularity = torch.sqrt(torch.tensor(self.num_bins)).int().item()
            increment = 1.11/granularity
            self_obs = states[0].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[..., 0, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[..., 1, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0 + granularity*y_1).int()
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            return y_out
        elif self.binning == "y_pos_door":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 4).int().item()
            increment = 1.11/granularity
            self_obs = states[0].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[..., 0, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[..., 1, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            door_status = door_obs[..., 0, 2] + 2*door_obs[..., 1, 2]
            #print(door_status)
            return (y_0 + granularity*y_1 + granularity*granularity*door_status).int()
        elif self.binning == "x_y_pos_door":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 64).int().item()
            increment = 1.33/granularity
            print("States 0 shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, self.num_agents, 9)
            y_0 = torch.clamp(self_obs[..., 0, 3], 0, 1.3) // increment # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[..., 1, 3], 0, 1.3) // increment # Granularity of 0.01 on the y
            x_0 = (torch.clamp(self_obs[..., 0, 2], -0.2, 0.2) + 0.2) // 0.101 #
            x_1 = (torch.clamp(self_obs[..., 1, 2], -0.2, 0.2) + 0.2) // 0.101 #
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(-1, self.num_worlds, self.num_agents, 3)
            door_status = door_obs[..., 0, 2] + 2*door_obs[..., 1, 2]
            print("Binning shapes", y_0.shape, y_1.shape, x_0.shape, x_1.shape, door_status.shape)
            return (y_0 + granularity*y_1 + granularity*granularity*x_0 + granularity*granularity*4*x_1 + granularity*granularity*4*4*door_status).int()
        elif self.binning == "y_pos_door_block":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 40).int().item()
            increment = 1.11/granularity
            self_obs = states[0].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[..., 0, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[..., 1, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            door_status = door_obs[..., 0, 2] + 2*door_obs[..., 1, 2]
            # Also bin block_pos since we want the blocks on the doors
            #print(states[2].shape)
            # Maybe for now average distance of the blocks from each agent
            block_obs = states[2].view(args.steps_per_update, self.num_worlds, self.num_agents, -1, 3)
            block_val = (block_obs[..., 2].mean(dim=2).sum(dim=2)*8).int() % 10
            #print("Block val", block_val.mean())
            #print(door_status)
            return (block_val*(granularity*granularity*4) + door_status*(granularity*granularity) + (y_0 + granularity*y_1)).int()
        else:
            raise NotImplementedError

    # Step 4: Map encountered states to bins
    def map_states_to_bins(self, states):
        # Apply binning function to define bin for new states
        bins = self.apply_binning_function(states)
        if torch.any(bins > self.num_bins):
            # throw error
            raise ValueError("Bin value too large")
        # Now return the binning of all states
        return bins

    def update_bin_steps(self, bins, prev_bins):
        #print(self.bin_steps[bins].shape, self.bin_steps[prev_bins].shape)
        #self.bin_steps[bins] = torch.minimum(self.bin_steps[bins], self.bin_steps[prev_bins] + 1)
        for i in range(1, args.steps_per_update):
            self.bin_steps[prev_bins[-i]] = torch.minimum(self.bin_steps[prev_bins[-i]], self.bin_steps[bins[-i]] + 1)

    # Step 5: Update archive
    def update_archive(self, bins, scores):
        # For each unique bin, update count in archive and update best score
        # At most can increase bin count by 1 in a single step...
        new_bin_counts = (torch.bincount(bins, minlength=self.num_bins) > 0).int()
        # Set the checkpoint for each bin to the latest
        #print(self.checkpoints)
        chosen_checkpoints = self.bin_count[bins] % self.num_checkpoints
        self.bin_count += new_bin_counts
        #print(chosen_checkpoints)
        #print(bins)
        self.bin_checkpoints[[bins, chosen_checkpoints]] = self.checkpoints # Will this have double-writes? Yes, shouldn't matter
        #self.bin_score[bins] = torch.maximum(self.bin_score[bins], scores)
        return None

    # Compute best score from archive
    def compute_best_score(self):
        return torch.max(self.state_score)
    
    # Learning callback
    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        skip_log = False
        if update_id != 1 and update_id % 10 != 0:
            skip_log = True

        ppo = update_results.ppo_stats

        if not skip_log:
            # Only log stuff from the worlds where we're not setting state
            desired_samples = int(self.num_worlds*args.new_frac)*self.num_agents
            print("Desired samples", desired_samples)
            with torch.no_grad():
                print(update_results.rewards.shape)
                reward_mean = update_results.rewards[:,desired_samples:].mean().cpu().item()
                reward_min = update_results.rewards[:,desired_samples:].min().cpu().item()
                reward_max = update_results.rewards[:,desired_samples:].max().cpu().item()

                done_count = (update_results.dones[:,desired_samples:] == 1.0).sum()
                return_mean, return_min, return_max = 0, 0, 0
                print("Update results shape", update_results.returns.shape)
                if done_count > 0:
                    print("Update results shape", update_results.returns[update_results.dones == 1.0].shape)
                    return_mean = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].mean().cpu().item()
                    return_min = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].min().cpu().item()
                    return_max = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].max().cpu().item()

                # compute visits to second and third room
                print("Update results shape", update_results.obs[0].shape, update_results.obs[3].shape)
                second_room_count = (update_results.obs[0][...,3] > 0.34)[:,:,desired_samples:].float().mean()
                third_room_count = (update_results.obs[0][...,3] > 0.67)[:,:,desired_samples:].float().mean()
                exit_count = (update_results.obs[0][...,3] > 1.01)[:,:,desired_samples:].float().mean()
                door_count = (update_results.obs[3][...,2] > 0.5)[:,:,desired_samples:].float().mean()

                second_room_count_unfiltered = (update_results.obs[0][...,3] > 0.34).float().mean()
                third_room_count_unfiltered = (update_results.obs[0][...,3] > 0.67).float().mean()
                exit_count_unfiltered = (update_results.obs[0][...,3] > 1.01).float().mean()
                door_count_unfiltered = (update_results.obs[3][...,2] > 0.5).float().mean()

                value_mean = update_results.values[:,desired_samples:].mean().cpu().item()
                value_min = update_results.values[:,desired_samples:].min().cpu().item()
                value_max = update_results.values[:,desired_samples:].max().cpu().item()

                advantage_mean = update_results.advantages[:,desired_samples:].mean().cpu().item()
                advantage_min = update_results.advantages[:,desired_samples:].min().cpu().item()
                advantage_max = update_results.advantages[:,desired_samples:].max().cpu().item()

                vnorm_mu = 0
                vnorm_sigma = 0
                if normalize_values:
                    vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
                    vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

            print(f"\nUpdate: {update_id}")
            print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
            print()
            print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
            print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
            print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
            print(f"    Returns          => Avg: {return_mean}, max: {return_max}")
            print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

            # Log average steps to end from known bins
            avg_steps = self.bin_steps[self.bin_count > 0].float().mean()

            # Add all this to wandb
            wandb.log({
                "update_id": update_id,
                "loss": ppo.loss,
                "action_loss": ppo.action_loss,
                "value_loss": ppo.value_loss,
                "entropy_loss": ppo.entropy_loss,
                "reward_mean": reward_mean, 
                "reward_max": reward_max,
                "returns_mean": return_mean,
                "returns_max": return_max,
                "done_count": done_count,
                "second_room_count": second_room_count,
                "third_room_count": third_room_count,
                "exit_count": exit_count,
                "door_count": door_count,
                "second_room_count_unfiltered": second_room_count_unfiltered,
                "third_room_count_unfiltered": third_room_count_unfiltered,
                "exit_count_unfiltered": exit_count_unfiltered,
                "door_count_unfiltered": door_count_unfiltered,
                "vnorm_mu": vnorm_mu,
                "steps_to_end": avg_steps,
                "max_progress": self.max_progress,
                }
            )

            if self.profile_report:
                print()
                print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
                print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
                profile.report()

            if update_id % 100 == 0:
                learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")

        #if update_id % 5 != 0:
        #    return

        # Now do the go-explore stuff
        goExplore.max_progress = max(goExplore.max_progress, min(update_results.obs[0][..., 0, 3].max(), update_results.obs[0][..., 1, 3].max()))
        if goExplore.max_progress > 1.01:
            exit_bins = goExplore.map_states_to_bins(goExplore.obs)[0,:][(goExplore.obs[0][...,3] > 1.01).view(goExplore.num_worlds, goExplore.num_agents).all(dim=1)]
            # Set exit path length to 0 for exit bins
            goExplore.bin_steps[exit_bins] = 0
            print("Exit bins", torch.unique(exit_bins).shape)
            #writer.add_scalar("charts/exit_path_length", goExplore.bin_steps[exit_bins].float().mean(), global_step)

        # Update archive from rollout
        #new_bins = self.map_states_to_bins(self.obs)
        all_bins = self.map_states_to_bins(update_results.obs) # num_timesteps * num_worlds
        self.update_bin_steps(all_bins[1:], all_bins[:-1])
        new_bins = all_bins[-1]
        # Update archive
        #print(self.curr_returns.shape)
        #print(self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).shape)
        self.curr_returns += self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1)
        self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
        #print(self.dones.shape)
        #print(self.dones)
        self.curr_returns *= (1 - 0.5*self.dones.view(self.num_worlds,self.num_agents).sum(dim=1)) # Account for dones, this is working!
        #print("Max return", torch.max(self.curr_returns), self.worlds.obs[torch.argmax(self.curr_returns)])
        self.update_archive(new_bins, self.curr_returns)
        # Set new state, go to state
        states = self.select_state()
        self.go_to_state(states)

# Maybe we can just write Go-Explore as a callback

# Before training: initialize archive
# During training:
#   1. Run rollout + PPO
#   2. Update archive from rollout
#   3. Run diagnostic if desired
#   4. Select next state for next rollout

# Now run the train loop from the other script
# Create GoExplore object from args
goExplore = GoExplore(args.num_worlds, dev)

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

run = wandb.init(
    # Set the project where this run will be logged
    project="escape-room-ppo-go",
    # Track hyperparameters and run metadata
    config=args
)

train(
    dev,
    SimInterface(
            step = lambda: goExplore.worlds.step(),
            obs = goExplore.obs,
            actions = goExplore.actions,
            dones = goExplore.dones,
            rewards = goExplore.rewards,
            resets = goExplore.resets,
            checkpoints = goExplore.checkpoints,
            checkpoint_resets = goExplore.checkpoint_resets
    ),
    TrainConfig(
        run_name = args.run_name,
        num_updates = args.num_updates,
        steps_per_update = args.steps_per_update,
        num_bptt_chunks = args.num_bptt_chunks,
        lr = args.lr,
        gamma = args.gamma,
        gae_lambda = 0.95,
        ppo = PPOConfig(
            num_mini_batches=1,
            clip_coef=0.2,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_loss_coef,
            max_grad_norm=0.5,
            num_epochs=2,
            clip_value_loss=args.clip_value_loss,
            no_advantages=args.no_advantages,
        ),
        value_normalizer_decay = args.value_normalizer_decay,
        mixed_precision = args.fp16,
        normalize_advantages = normalize_advantages,
        normalize_values = normalize_values,
    ),
    goExplore.policy,
    goExplore,
    restore_ckpt
)