import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
import math

import madrona_learn
from madrona_learn import (
    ActorCritic, BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer, ObservationsCaster,
)

from madrona_learn.models import (
    LayerNorm,
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
)
from madrona_learn.rnn import LSTM

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

def pytorch_initializer():
    scale = 2 / (1 + math.sqrt(2) ** 2)
    return jax.nn.initializers.variance_scaling(
        scale, mode='fan_in', distribution='normal')

class PrefixCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, self_ob = obs.pop('self')
        obs, steps_remaining = obs.pop('stepsRemaining')
        obs, lidar = obs.pop('lidar')
        obs, agent_id = obs.pop('agentID')

        #lidar = nn.Conv(
        #        features=1,
        #        kernel_size=(lidar.shape[-2],),
        #        padding='CIRCULAR',
        #        dtype=self.dtype,
        #    )(lidar)
        lidar = lidar.reshape(*lidar.shape[0:-2], -1)

        self_ob = jnp.concatenate([
                self_ob,
                steps_remaining,
                lidar,
                agent_id,
            ], axis=-1)

        return obs.copy({
            'self': self_ob, 
        })
        

class PrefixCommonSimpleAdapter(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        xs = PrefixCommon(dtype = self.dtype)(obs, train)

        xs, self = xs.pop('self')
        xs = jax.tree_map(
            lambda x: x.reshape(*x.shape[:-2], -1), xs)

        xs = xs.copy({'self': self})

        flattened, _ = jax.tree_util.tree_flatten(xs)

        return jnp.concatenate(flattened, axis=-1)


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


class PolicyLSTM(nn.Module):
    num_hidden_channels: int
    num_layers: int
    dtype: jnp.dtype

    def setup(self):
        self.lstm = LSTM(
            num_hidden_channels = self.num_hidden_channels,
            num_layers = self.num_layers,
            dtype = self.dtype,
        )

        self.layernorm = LayerNorm(dtype=self.dtype)

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        return self.layernorm(self.lstm(cur_hiddens, x, train))

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.layernorm(
            self.lstm.sequence(start_hiddens, seq_ends, seq_x, train))

def make_policy(dtype, use_simple_policy):

    if use_simple_policy:
        prefix = PrefixCommonSimpleAdapter(dtype)
        #prefix = ProcessObsMLP(dtype)
        encoder = madrona_learn.BackboneEncoder(
            net = MLP(
                num_channels = 256,
                num_layers = 3,
                dtype = dtype,
                weight_init = pytorch_initializer(),
            ),
            #rnn = LSTM(
            #    num_hidden_channels = 256,
            #    num_layers = 1,
            #    dtype = dtype,
            #),
        )
    else:
        prefix = PrefixCommon(dtype)
        encoder = madrona_learn.BackboneEncoder(
            net = EntitySelfAttentionNet(
                num_embed_channels = 128,
                num_heads = 4,
                dtype = dtype,
            ),
            #rnn = PolicyLSTM(
            #    num_hidden_channels = 256,
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

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
    )

    #obs_preprocess = ObservationsCaster(dtype=dtype)
    #obs_preprocess = None

    return policy, obs_preprocess
