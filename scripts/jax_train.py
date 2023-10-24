import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial

import madrona_escape_room

import madrona_learn
from madrona_learn import (
    TrainConfig, CustomMetricConfig, PPOConfig,
)

from madrona_learn.rnn import LSTM
from jax_policy import make_policy

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

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

arg_parser.add_argument('--pbt-ensemble-size', type=int, default=1)
arg_parser.add_argument('--pbt-history-len', type=int, default=1)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-report', action='store_true')

args = arg_parser.parse_args()

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_step, init_sim_data = sim.jax(jax_gpu)

def metrics_cb(metrics, epoch, mb, train_state):
    return metrics

def host_cb(update_idx, metrics, train_state_mgr):
    print(f"Update: {update_idx}")

    metrics.pretty_print()
    vnorm_mu = train_state_mgr.train_states.value_normalize_stats['mu'][0][0]
    vnorm_sigma = train_state_mgr.train_states.value_normalize_stats['sigma'][0][0]
    print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma: .3e}")
    print()

    train_state_mgr.save(update_idx, f"{args.ckpt_dir}/{update_idx}")

    return ()

def iter_cb(update_idx, update_time, metrics, train_state_mgr):
    cb = partial(jax.experimental.io_callback, host_cb, ())
    noop = lambda *args: ()

    update_id = update_idx + 1
    lax.cond(jnp.logical_or(update_id == 1, update_id % 10 == 0), cb, noop,
             update_id, metrics, train_state_mgr)
    #cb(update_id, metrics, train_state_mgr)

dev = jax.devices()[0]

cfg = TrainConfig(
    num_worlds = args.num_worlds,
    team_size = 2,
    num_teams = 1,
    num_updates = args.num_updates,
    steps_per_update = args.steps_per_update,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = args.lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        num_mini_batches=1,
        clip_coef=0.2,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_loss_coef,
        max_grad_norm=0.5,
        num_epochs=2,
        clip_value_loss=args.clip_value_loss,
    ),
    value_normalizer_decay = 0.999,
    mixed_precision = args.fp16,
    seed = 5,
    pbt_ensemble_size = args.pbt_ensemble_size,
    pbt_history_len = args.pbt_history_len,
)

policy = make_policy(jnp.float16 if args.fp16 else jnp.float32, True)

madrona_learn.train(dev, cfg, sim_step, init_sim_data, policy,
    iter_cb, CustomMetricConfig(register_metrics = lambda metrics: metrics))

del sim
