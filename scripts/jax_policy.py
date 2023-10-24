import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial

import madrona_learn
from madrona_learn import (
    ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    EMANormalizeTree,
    MLP,
    EgocentricSelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

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

class ProcessObsMLP(nn.Module):
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

        obs, steps_remaining = obs.pop('stepsRemaining')
        steps_remaining = steps_remaining / 200

        processed_obs = obs.copy({
            'stepsRemaining': steps_remaining,
        })

        flattened, _ = jax.tree_util.tree_flatten(processed_obs)

        return jnp.concatenate(flattened, axis=-1)

def make_policy(dtype, use_simple_policy):

    if use_simple_policy:
        prefix = ProcessObsMLP(dtype)
        encoder = madrona_learn.BackboneEncoder(
            net = MLP(
                num_channels = 256,
                num_layers = 3,
                dtype = dtype,
            ),
            #rnn = LSTM(
            #    hidden_channels = 256,
            #    num_layers = 1,
            #    dtype = dtype,
            #),
        )
    else:
        prefix = ProcessObsCommon(dtype)
        encoder = madrona_learn.BackboneEncoder(
            net = EgocentricSelfAttentionNet(
                num_embed_channels = 64,
                num_heads = 2,
                dtype = dtype,
            ),
            #rnn = LSTM(
            #    hidden_channels = 256,
            #    num_layers = 1,
            #    dtype = dtype,
            #),
        )

    backbone = madrona_learn.BackboneShared(
        prefix = prefix,
        encoder = encoder,
    )

    policy = ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            actions_num_buckets = [4, 8, 5, 2],
            dtype = dtype,
        ),
        critic = DenseLayerCritic(dtype=dtype),
    )

    return policy
