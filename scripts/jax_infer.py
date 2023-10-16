import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial

import madrona_escape_room

import madrona_learn
from madrona_learn import InferConfig

from jax_policy import make_policy

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-policies', type=int, default=1)
arg_parser.add_argument('--num-steps', type=int, default=200)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--action-dump-path', type=str)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_step, init_sim_data = sim.jax(jax_gpu)

if args.action_dump_path:
    action_log = open(args.action_dump_path, 'wb')
else:
    action_log = None

def host_cb(actions, action_probs, values, dones, rewards):
    print(actions)
    print(action_probs[0][0, 0], action_probs[0][0, 1])
    print(action_probs[1][0, 0], action_probs[1][0, 1])
    print(action_probs[2][0, 0], action_probs[2][0, 1])
    print(action_probs[3][0, 0], action_probs[3][0, 1])
    print(rewards)

    if action_log:
        actions.tofile(action_log)

    return ()

def iter_cb(actions, action_probs, values, dones, rewards):
    cb = partial(jax.experimental.io_callback, host_cb, ())

    cb(actions, action_probs, values, dones, rewards)

dev = jax.devices()[0]

policy = make_policy(jnp.float16, True)

infer_cfg = InferConfig(
    num_steps = args.num_steps,
    num_policies = args.num_policies,
    mixed_precision = True,
)

madrona_learn.infer(dev, infer_cfg, sim_step, init_sim_data,
    policy, iter_cb, args.ckpt_path)

del sim
