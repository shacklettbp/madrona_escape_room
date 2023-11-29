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
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--seed', type=int, required=True)
arg_parser.add_argument('--use-logging', action='store_true')

# World gen args
arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--start-in-discovered-rooms', action='store_true')
arg_parser.add_argument('--reward-mode', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

# Learning args
arg_parser.add_argument('--exploration-steps', type=int, required=True)
arg_parser.add_argument('--binning', type=str, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--num-bins', type=int, required=True)
arg_parser.add_argument('--num-checkpoints', type=int, default=1)

# Diagnostic args
arg_parser.add_argument('--bin-diagnostic', action='store_true')
arg_parser.add_argument('--seeds-per-checkpoint', type=int, default=16)


args = arg_parser.parse_args()

sim_flags = SimFlags.Default
if args.use_fixed_world:
    sim_flags |= SimFlags.UseFixedWorld
if args.start_in_discovered_rooms:
    sim_flags |= SimFlags.StartInDiscoveredRooms

reward_mode = getattr(RewardMode, args.reward_mode)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

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
    def __init__(self, num_worlds, exploration_steps, device):
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
        self.num_exploration_steps = exploration_steps
        self.binning = args.binning
        self.num_bins = args.num_bins # We can change this later
        self.num_checkpoints = args.num_checkpoints
        self.device = device
        self.checkpoint_score = torch.zeros(self.num_bins, self.num_checkpoints, device=device)
        self.bin_count = torch.zeros(self.num_bins, device=device).int()
        self.max_return = 0

        self.obs, num_obs_features = setup_obs(self.worlds)
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.checkpoints = self.worlds.checkpoint_tensor().to_torch()
        self.checkpoint_resets = self.worlds.checkpoint_reset_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()
        self.bin_checkpoints = torch.zeros((self.num_bins, self.num_checkpoints, self.checkpoints.shape[-1]), device=device, dtype=torch.uint8)

        self.actions_num_buckets = [4, 8, 5, 2]
        self.action_space = Box(-float('inf'),float('inf'),(sum(self.actions_num_buckets),))


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
        sampled_bins = valid_bins[torch.multinomial(weights, num_samples=self.num_worlds, replacement=True).type(torch.int)]
        # Sample states from bins: either sample first occurrence in each bin (what's in the paper), or something better...
        # Need the last checkpoint for each bin
        chosen_checkpoint = self.bin_count[sampled_bins] % self.num_checkpoints
        self.curr_returns[:] = self.checkpoint_score[[sampled_bins, chosen_checkpoint]]
        return self.bin_checkpoints[[sampled_bins, chosen_checkpoint]]

    # Step 2: Go to state
    # Input: states, worlds
    # Logic: For each state, set world to state
    # Output: None
    def go_to_state(self, states):
        # Run checkpoint-restoration here
        self.checkpoint_resets[:, 0] = 1
        self.checkpoints[:] = states
        self.worlds.step()
        self.obs[5][:] = 200
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
            incrememnt = 1.1/granularity
            self_obs = states[0].view(self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[:, 0, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[:, 1, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            y_out = (y_0 + granularity*y_1).int()
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            return y_out
        elif self.binning == "y_pos_door":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 4).int().item()
            incrememnt = 1.1/granularity
            self_obs = states[0].view(self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[:, 0, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[:, 1, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(self.num_worlds, self.num_agents, -1)
            door_status = door_obs[:, 0, 2] + 2*door_obs[:, 1, 2]
            #print(door_status)
            return (y_0 + granularity*y_1 + granularity*granularity*door_status).int()
        elif self.binning == "x_y_pos_door":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 64).int().item()
            incrememnt = 1.1/granularity
            self_obs = states[0].view(self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[:, 0, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[:, 1, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            x_0 = (torch.clamp(self_obs[:, 0, 2], -0.2, 0.2) + 0.2) // 0.1 #
            x_1 = (torch.clamp(self_obs[:, 1, 2], -0.2, 0.2) + 0.2) // 0.1 #
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(self.num_worlds, self.num_agents, -1)
            door_status = door_obs[:, 0, 2] + 2*door_obs[:, 1, 2]
            #print(door_status)
            return (y_0 + granularity*y_1 + granularity*granularity*x_0 + granularity*granularity*4*x_1 + granularity*granularity*4*4*door_status).int()
        elif self.binning == "y_pos_door_block":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 40).int().item()
            incrememnt = 1.1/granularity
            self_obs = states[0].view(self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[:, 0, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[:, 1, 3], 0, 1.1) // incrememnt # Granularity of 0.01 on the y
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(self.num_worlds, self.num_agents, -1)
            door_status = door_obs[:, 0, 2] + 2*door_obs[:, 1, 2]
            # Also bin block_pos since we want the blocks on the doors
            #print(states[2].shape)
            # Maybe for now average distance of the blocks from each agent
            block_obs = states[2].view(self.num_worlds, self.num_agents, -1, 3)
            block_val = (block_obs[:, :, :, 2].mean(dim=1).sum(dim=1)*8).int() % 10
            #print("Block val", block_val.mean())
            #print(door_status)
            return (block_val*(granularity*granularity*4) + door_status*(granularity*granularity) + (y_0 + granularity*y_1)).int()
        else:
            raise NotImplementedError

    # Step 4: Map encountered states to bins
    def map_states_to_bins(self, states):
        # Apply binning function to define bin for new states
        bins = self.apply_binning_function(states)
        # Now return the binning of all states
        return bins

    # Step 5: Update archive
    def update_archive(self, bins, scores):
        # For each unique bin, update count in archive and update best score
        # At most can increase bin count by 1 in a single step...
        new_bin_counts = (torch.bincount(bins, minlength=self.num_bins) > 0).int()
        self.bin_count += new_bin_counts
        # Set the checkpoint for each bin to the latest
        #print(self.checkpoints)
        chosen_checkpoints = self.bin_count[bins] % self.num_checkpoints
        #print(chosen_checkpoints)
        #print(bins)
        self.bin_checkpoints[[bins, chosen_checkpoints]] = self.checkpoints # Will this have double-writes? Yes, shouldn't matter
        #self.bin_score[bins] = torch.maximum(self.bin_score[bins], scores)
        return None

    # Compute best score from archive
    def compute_best_score(self):
        return torch.max(self.state_score)

# Run training loop
def train(args):
    # Create GoExplore object from args
    goExplore = GoExplore(args.num_worlds, args.exploration_steps, dev)
    # Set up wandb
    run_name = f"go_explore_{int(time.time())}"
    if args.use_logging:
        wandb.init(
            project="escape_room_go_explore",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    best_score = 0
    start_time = time.time()
    # Before starting, initialize the first state as an option
    #print("About to initialize archive")
    start_bin = goExplore.map_states_to_bins(goExplore.obs)
    goExplore.update_archive(start_bin, torch.zeros(args.num_worlds, device=dev))
    for i in range(args.num_steps):
        # Step 1: Select state from archive
        states = goExplore.select_state()
        # Step 2: Go to state
        goExplore.go_to_state(states)
        # Step 3: Explore from state
        goExplore.explore_from_state()
        # Compute best score from archive
        # best_score = max(best_score, goExplore.compute_best_score())
        #print(goExplore.max_return)
        # Log the step
        global_step = (i + 1)*args.num_worlds
        if args.use_logging and i % 10 == 0:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/best_score", goExplore.max_return, global_step)
            # Compute number of unvisited and underexplored states from archive
            unvisited_bins = torch.sum(goExplore.bin_count == 0) # Need to fix this when num_states != num_bins
            # Log it all
            writer.add_scalar("charts/unvisited_bins", unvisited_bins, global_step)
            # Specific to escape room
            second_room_count = (goExplore.obs[0][...,3] > 0.34).sum()
            third_room_count = (goExplore.obs[0][...,3] > 0.67).sum()
            exit_count = (goExplore.obs[0][...,3] > 1.01).sum()
            door_count = (goExplore.obs[3][...,2] > 0.5).sum()
            writer.add_scalar("charts/second_room_count", second_room_count, global_step)
            writer.add_scalar("charts/third_room_count", third_room_count, global_step)
            writer.add_scalar("charts/exit_count", exit_count, global_step)
            writer.add_scalar("charts/door_count", door_count, global_step)
            self_obs = goExplore.obs[0].view(goExplore.num_worlds, goExplore.num_agents, -1)
            writer.add_scalar("charts/max_agent_0_progress", self_obs[:, 0, 3].max(), global_step)
            writer.add_scalar("charts/max_agent_1_progress", self_obs[:, 1, 3].max(), global_step)
        if args.bin_diagnostic:
            print("Hello there")
            # Step 1: Select random states from archive that have all checkpoints filled. We want to run k parallel trials from each checkpoint
            filled_bins = torch.nonzero(goExplore.bin_count >= args.num_checkpoints).flatten()
            required_bins = goExplore.num_worlds // (args.num_checkpoints * args.seeds_per_checkpoint)
            print("Required bins", required_bins, "Filled bins", len(filled_bins))
            if required_bins > len(filled_bins):
                print("Not enough bins to run all trials, skipping diagnostic")
                continue
            if (goExplore.obs[0][...,3] > 1.01).sum() == 0:
                print("No agents have reached the exit, skipping diagnostic")
                continue
            all_results = []
            for k in range(50): # Take 50 samples
                print(k)
                selected_bins = filled_bins[torch.multinomial(torch.ones(len(filled_bins)), num_samples=required_bins, replacement=False).type(torch.int)]
                goExplore.go_to_state(torch.repeat_interleave(goExplore.bin_checkpoints[selected_bins].view(-1, goExplore.bin_checkpoints.shape[-1]), args.seeds_per_checkpoint, dim=0)) # Should get all checkpoints from each selected bin
                results = []
                for i in range(10):
                    # Step 2: Take a step forward with random action on each world
                    goExplore.actions[:] = goExplore.generate_random_actions()
                    goExplore.worlds.step()
                    # Step 3: Compute bin distribution for each checkpoint
                    current_bins = goExplore.map_states_to_bins(goExplore.obs)
                    #print("Current bins", current_bins)
                    #print("Current bin count", torch.bincount(current_bins, minlength=goExplore.num_bins))
                    #print()
                    #print("After ", i, "steps")
                    num_bins = torch.nonzero(torch.bincount(current_bins, minlength=goExplore.num_bins)).flatten().shape[0]
                    #print("Nonzero bins", num_bins)
                    results.append(num_bins)
                    '''
                    spc = args.seeds_per_checkpoint
                    nc = args.num_checkpoints
                    bin_distribution = [torch.bincount(current_bins[spc*i:spc*(i+1)], minlength=goExplore.num_bins)/spc for i in range(nc * required_bins)]
                    # Step 4: Compare to other bin distributions from same starting bin
                    bin_distribution_vars = []
                    for j in range(required_bins):
                        # Compute variance in bincount for the same starting bin
                        print("Stack shape", torch.stack(bin_distribution[nc*j:nc*(j+1)]).shape)
                        print("Var shape", torch.var(torch.stack(bin_distribution[nc*j:nc*(j+1)]), dim=0).shape)
                        bin_distribution_vars.append(spc * torch.var(torch.stack(bin_distribution[nc*j:nc*(j+1)]), dim=0).sum()) # Multiply by spc to reverse CLT effect on Var
                    # Step 5: Log info
                    print("Bin distribution vars", torch.mean(torch.tensor(bin_distribution_vars, device=dev)))
                    '''
                all_results.append(torch.tensor(results, device=dev))
            all_results = torch.stack(all_results, dim=0).float().mean(dim=0)
            print("Average number of bins at each step", all_results)
            return

    # Return best score
    return best_score

train(args)
