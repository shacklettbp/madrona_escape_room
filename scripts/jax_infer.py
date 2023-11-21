import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
import numpy as np

import argparse
from functools import partial

import madrona_escape_room

import madrona_learn

from jax_policy import make_policy

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-policies', type=int, default=1)
arg_parser.add_argument('--num-steps', type=int, default=200)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--action-dump-path', type=str)
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    sim_flags = madrona_escape_room.SimFlags.Default,
    reward_mode = madrona_escape_room.RewardMode.OG,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_step, init_sim_data = sim.jax(jax_gpu)

if args.action_dump_path:
    action_log = open(args.action_dump_path, 'wb')
else:
    action_log = None

def host_cb(obs, actions, action_probs, values, dones, rewards):
    print(obs)

    print("Actions:", actions)

    print("Move Amount Probs")
    print(" ", np.array_str(action_probs[0][0], precision=2, suppress_small=True))
    print(" ", np.array_str(action_probs[0][1], precision=2, suppress_small=True))

    print("Move Angle Probs")
    print(" ", np.array_str(action_probs[1][0], precision=2, suppress_small=True))
    print(" ", np.array_str(action_probs[1][1], precision=2, suppress_small=True))

    print("Rotate Probs")
    print(" ", np.array_str(action_probs[2][0], precision=2, suppress_small=True))
    print(" ", np.array_str(action_probs[2][1], precision=2, suppress_small=True))

    print("Grab Probs")
    print(" ", np.array_str(action_probs[3][0], precision=2, suppress_small=True))
    print(" ", np.array_str(action_probs[3][1], precision=2, suppress_small=True))

    print("Rewards:", rewards)

    if action_log:
        actions.tofile(action_log)

    return ()

def iter_cb(step_data):
    cb = partial(jax.experimental.io_callback, host_cb, ())

    cb(step_data['obs'],
       step_data['actions'],
       step_data['action_probs'],
       step_data['values'],
       step_data['dones'],
       step_data['rewards'])

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

policy, obs_preprocess = make_policy(dtype, False)

madrona_learn.eval_ckpt(dev, args.ckpt_path, args.num_steps, sim_step,
    init_sim_data, policy, obs_preprocess, iter_cb, dtype)

del sim
