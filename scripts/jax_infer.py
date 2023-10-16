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
    ActorCritic, PPOConfig, InferConfig,
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    EMANormalizeTree,
    EgocentricSelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

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

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class ProcessObsCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs = jax.tree_map(lambda x: jnp.asarray(x, dtype=self.dtype), obs)

        jax.tree_map(lambda x: assert_valid_input(x), obs)

        obs = jax.tree_map(
            lambda o: o.reshape(o.shape[0], -1), obs)

        normalized_obs = EMANormalizeTree(0.99999)(obs, train)

        normalized_obs, lidar = normalized_obs.pop('lidar')

        lidar_processed = nn.Conv(
                features=1,
                kernel_size=(lidar.shape[-1],),
                padding='CIRCULAR',
                dtype=self.dtype,
            )(jnp.expand_dims(lidar, axis=-1))
        
        lidar_processed = lidar_processed.squeeze(axis=-1)
        lidar_processed = lidar

        return normalized_obs.copy({
            'lidar': lidar_processed
        })

def metrics_cb(metrics, epoch, mb, train_state):
    return metrics

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

def test3():
    dev = jax.devices()[0]

    encoder = madrona_learn.RecurrentBackboneEncoder(
        net = EgocentricSelfAttentionNet(
            num_embed_channels = 64,
            num_heads = 2,
            dtype = jnp.float16,
        ),
        rnn = LSTM(
            hidden_channels = 256,
            num_layers = 1,
            dtype = jnp.float16,
        ),
    )

    backbone = madrona_learn.BackboneShared(
        prefix = ProcessObsCommon(
            dtype = jnp.float16,
        ),
        encoder = encoder,
    )

    policy = ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            actions_num_buckets = [4, 8, 5, 2],
            dtype = jnp.float16,
        ),
        critic = DenseLayerCritic(dtype=jnp.float16),
    )

    infer_cfg = InferConfig(
        num_steps = args.num_steps,
        num_policies = args.num_policies,
        mixed_precision = True,
    )

    madrona_learn.infer(dev, infer_cfg, sim_step, init_sim_data,
        policy, iter_cb, args.ckpt_path)

test3()

del sim
