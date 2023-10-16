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
    ActorCritic, TrainConfig, CustomMetricConfig, PPOConfig,
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
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
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

def host_cb(update_idx, metrics, train_state_mgr):
    print(f"Update: {update_idx}")
    print(metrics)

    train_state_mgr.save(update_idx, f"{args.ckpt_dir}/{update_idx}")

    return ()

def iter_cb(update_idx, update_time, metrics, train_state_mgr):
    cb = partial(jax.experimental.io_callback, host_cb, ())
    noop = lambda *args: ()

    lax.cond(update_idx % 10 == 0, cb, noop,
             update_idx, metrics, train_state_mgr)

def test3():
    dev = jax.devices()[0]

    cfg = TrainConfig(
        num_worlds = args.num_worlds,
        num_updates = 5000,
        steps_per_update = 40,
        lr = 1e-4,
        algo = PPOConfig(
            num_epochs = 2,
            num_mini_batches = 1,
            clip_coef = 0.2,
            value_loss_coef = 0.5,
            entropy_coef = 0.01,
            max_grad_norm = 0.5,
        ), 
        num_bptt_chunks = 8,
        pbt_ensemble_size = args.num_policies,
        gamma = 0.998,
        gae_lambda = 0.95,
        value_normalizer_decay = 0.999,
        team_size = 2,
        seed = 5,
        mixed_precision=True,
    )

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

    madrona_learn.train(dev, cfg, sim_step, init_sim_data, policy,
        iter_cb, CustomMetricConfig(cb=metrics_cb, custom_metrics=[]))

test3()

del sim
