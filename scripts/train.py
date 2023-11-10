import madrona_escape_room

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

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--reward-mode', type=str, required=True)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-report', action='store_true')

arg_parser.add_argument('--no-value-norm', action='store_true')
arg_parser.add_argument('--no-advantage-norm', action='store_true')

arg_parser.add_argument('--no-advantages', action='store_true')
arg_parser.add_argument('--value-normalizer-decay', type=float, default=0.999)

args = arg_parser.parse_args()

normalize_values = not args.no_value_norm
normalize_advantages = not args.no_advantage_norm

run = wandb.init(
    # Set the project where this run will be logged
    project="escape-room-lugia",
    # Track hyperparameters and run metadata
    config=args
)

class LearningCallback:
    def __init__(self, ckpt_dir, profile_report):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        if update_id != 1 and  update_id % 10 != 0:
            return

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_mean = update_results.rewards.mean().cpu().item()
            reward_min = update_results.rewards.min().cpu().item()
            reward_max = update_results.rewards.max().cpu().item()

            done_count = (update_results.dones == 1.0).sum()
            return_mean, return_min, return_max = 0, 0, 0
            if done_count > 0:
                return_mean = update_results.returns[update_results.dones == 1.0].mean().cpu().item()
                return_min = update_results.returns[update_results.dones == 1.0].min().cpu().item()
                return_max = update_results.returns[update_results.dones == 1.0].max().cpu().item()

            # compute visits to second and third room
            second_room_count = (update_results.obs[0][:,:,3] > 0.34).sum()
            third_room_count = (update_results.obs[0][:,:,3] > 0.67).sum()
            exit_count = (update_results.obs[0][:,:,3] > 1.01).sum()
            door_count = (update_results.obs[3][:,:,2] > 0.5).sum()

            value_mean = update_results.values.mean().cpu().item()
            value_min = update_results.values.min().cpu().item()
            value_max = update_results.values.max().cpu().item()

            advantage_mean = update_results.advantages.mean().cpu().item()
            advantage_min = update_results.advantages.min().cpu().item()
            advantage_max = update_results.advantages.max().cpu().item()

            bootstrap_value_mean = update_results.bootstrap_values.mean().cpu().item()
            bootstrap_value_min = update_results.bootstrap_values.min().cpu().item()
            bootstrap_value_max = update_results.bootstrap_values.max().cpu().item()

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
        print(f"    Bootstrap Values => Avg: {bootstrap_value_mean: .3e}, Min: {bootstrap_value_min: .3e}, Max: {bootstrap_value_max: .3e}")
        print(f"    Returns          => Avg: {return_mean}, max: {return_max}")
        print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

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
            "vnorm_mu": vnorm_mu,
            }
        )

        if self.profile_report:
            print()
            print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
            print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
            profile.report()

        if update_id % 100 == 0:
            learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")

reward_mode = getattr(madrona_escape_room.RewardMode, args.reward_mode)

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    use_fixed_world = args.use_fixed_world,
    reward_mode = reward_mode,
)

use_warm_up = True
if use_warm_up:
    steps_so_far = 0
    warm_up = 32
    while steps_so_far < 200:
        for i in range(warm_up - 1):
            sim.step()
        resets = sim.reset_tensor().to_torch().view(-1)
        total_envs = resets.shape[0]
        reset_min = (steps_so_far / 200)
        reset_max = ((steps_so_far + warm_up) / 200)
        resets[(int)(reset_min * total_envs):(int)(reset_max * total_envs)] = 1
        print("Steps so far", steps_so_far)
        print("Max length", 200)
        print("Resetting", (int)(reset_min * total_envs), (int)(reset_max * total_envs))
        sim.step()
        steps_so_far += warm_up

ckpt_dir = Path(args.ckpt_dir)

learning_cb = LearningCallback(ckpt_dir, args.profile_report)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir.mkdir(exist_ok=True, parents=True)

obs, num_obs_features = setup_obs(sim)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

train(
    dev,
    SimInterface(
            step = lambda: sim.step(),
            obs = obs,
            actions = actions,
            dones = dones,
            rewards = rewards,
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
            num_epochs=1,
            clip_value_loss=args.clip_value_loss,
            no_advantages=args.no_advantages,
        ),
        value_normalizer_decay = args.value_normalizer_decay,
        mixed_precision = args.fp16,
        normalize_advantages = normalize_advantages,
        normalize_values = normalize_values,
    ),
    policy,
    learning_cb,
    restore_ckpt
)
