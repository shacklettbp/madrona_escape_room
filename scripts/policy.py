from madrona_escape_room_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder,
    RecurrentBackboneEncoder,
)

from madrona_escape_room_learn.models import (
    CNN, MLP, LinearLayerDiscreteActor, LinearLayerCritic,
    DenseLayerDiscreteActor, DenseLayerCritic,
)

from madrona_escape_room_learn.rnn import LSTM

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

def setup_obs(sim, raw_pixels=False):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    partner_obs_tensor = sim.partner_observations_tensor().to_torch()
    room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
    door_obs_tensor = sim.door_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

    if raw_pixels:
        rgb_tensor = sim.rgb_tensor().to_torch()     # tensor shape = B (16384), W (64), H (64), C (4)
        depth_tensor = sim.depth_tensor().to_torch() # tensor shape = B (16384), W (64), H (64), C (1)

    A = 2
    N = self_obs_tensor.shape[0] // A

    # Add in an agent ID tensor
    id_tensor = torch.arange(A).float()
    if A > 1:
        id_tensor = id_tensor / (A - 1)

    id_tensor = id_tensor.to(device=self_obs_tensor.device)
    id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(N * A, 1)

    if not raw_pixels:
        obs_tensors = [
            self_obs_tensor,        # size = torch.Size([16384, 8])    = B (16384), # of attributes in self ob (8)
            partner_obs_tensor,     # size = torch.Size([16384, 1, 3]) = B (16384), # of partners (1), # of attributes in partner ob (3)
            room_ent_obs_tensor,    # size = torch.Size([16384, 6, 3]) = B (16384), # of room entities (6), # of attributes in room ent ob (3)
            door_obs_tensor,        # size = torch.Size([16384, 1, 3]) = B (16384), # of doors (1), # of attributes in door ob (3)
            lidar_tensor,           # size = torch.Size([16384, 30, 2])= B (16384), # of lidar rays (30), # of attributes in lidar ray (2 = depth, intensity)
            steps_remaining_tensor, # size = torch.Size([16384, 1])    = B (16384), # of attributes in steps remaining (1)
            id_tensor,              # size = torch.Size([16384, 1])    = B (16384), # of attributes in agent ID (1)
        ]

        num_obs_features = 0
        for tensor in obs_tensors:
            num_obs_features += math.prod(tensor.shape[1:])
        # num_obs_features = 8 + 3 + 18 + 3 + 60 + 1 + 1 = 94
            
        return obs_tensors, num_obs_features
        
    else:
        # raw pixels
        obs_tensors = [
            rgb_tensor,             # size = torch.Size([16384, 64, 64, 4]) = B (16384), W (64), H (64), C (4)
            depth_tensor,           # size = torch.Size([16384, 64, 64, 1]) = B (16384), W (64), H (64), C (1)
        ]

        num_channels = 4
        
        return obs_tensors, num_channels

def process_obs(self_obs, partner_obs, room_ent_obs,
                door_obs, lidar, steps_remaining, ids):
    assert(not torch.isnan(self_obs).any())
    assert(not torch.isinf(self_obs).any())

    assert(not torch.isnan(partner_obs).any())
    assert(not torch.isinf(partner_obs).any())

    assert(not torch.isnan(room_ent_obs).any())
    assert(not torch.isinf(room_ent_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(steps_remaining).any())
    assert(not torch.isinf(steps_remaining).any())

    return torch.cat([
        self_obs.view(self_obs.shape[0], -1),
        partner_obs.view(partner_obs.shape[0], -1),
        room_ent_obs.view(room_ent_obs.shape[0], -1),
        door_obs.view(door_obs.shape[0], -1),
        lidar.view(lidar.shape[0], -1),
        steps_remaining.float() / 200,
        ids,
    ], dim=1)

def process_pixels(rgb, depth):
    assert(not torch.isnan(rgb).any())
    assert(not torch.isinf(rgb).any())

    # convert depth nan or inf to 0
    depth[torch.isnan(depth)] = 0
    depth[torch.isinf(depth)] = 0

    assert(not torch.isnan(depth).any())
    assert(not torch.isinf(depth).any())

    # cast to half precision
    rgb = rgb.to(torch.float16)
    depth = depth.to(torch.float16)

    obs = torch.cat([
        rgb[..., 0:3],
        depth,
    ], dim=-1).to(torch.float16)

    CNN_net = CNN(in_channels = obs.shape[-1]).to(obs.device).to(torch.float16)
    return CNN_net(obs).to(torch.float16)

def make_policy(dim_info, num_channels, separate_value, raw_pixels=False):
    if raw_pixels:
        encoder = RecurrentBackboneEncoder(
            net = MLP(input_dim = num_channels * dim_info,
                      num_channels = num_channels,
                      num_layers = 1),
            rnn = LSTM(in_channels = num_channels,
                       hidden_channels = num_channels,
                       num_layers = 1),
        )

        backbone = BackboneShared(
            process_obs = process_pixels,
            encoder = encoder,
        )

        return ActorCritic(
            backbone = backbone,
            actor = DenseLayerDiscreteActor(
                [4, 8, 5, 2],
                num_channels,
            ),
            critic = DenseLayerCritic(num_channels),
        )
    
    else:
        encoder = BackboneEncoder(
            net = MLP(
                input_dim = dim_info,
                num_channels = num_channels,
                num_layers = 3,
            ),
        )

        if separate_value:
            backbone = BackboneSeparate(
                process_obs = process_obs,
                actor_encoder = encoder,
                critic_encoder = RecurrentBackboneEncoder(
                    net = MLP(
                        input_dim = dim_info,
                        num_channels = num_channels,
                        num_layers = 2,
                    ),
                    rnn = LSTM(
                        in_channels = num_channels,
                        hidden_channels = num_channels,
                        num_layers = 1,
                    ),
                )
            )
        else:
            backbone = BackboneShared(
                process_obs = process_obs,
                encoder = encoder,
            )

        return ActorCritic(
            backbone = backbone,
            actor = LinearLayerDiscreteActor(
                [4, 8, 5, 2],
                num_channels,
            ),
            critic = LinearLayerCritic(num_channels),
        )