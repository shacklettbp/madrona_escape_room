import torch
import madrona_escape_room
from madrona_escape_room import RewardMode
import argparse
import time
from policy import setup_obs

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

reward_mode = getattr(RewardMode, "Dense1")

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    sim_flags = (int)(0),
    reward_mode = reward_mode,
    button_width = 1.3,
    door_width = 20.0 / 3.,
    reward_per_dist = 0.05,
    slack_reward = -0.005,
)

actions = sim.action_tensor().to_torch()
checkpoints = sim.checkpoint_tensor().to_torch()
checkpoint_resets = sim.checkpoint_reset_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
obs, _ = setup_obs(sim)

print("Start testing")

start = time.time()
for i in range(args.num_steps):
    print("Step", i)
    # Rotate worlds
    checkpoint_resets[:, 0] = 1
    # Copy checkpoints from first half of worlds to last half, but shifted by i
    checkpoints[args.num_worlds//2 + i:] = checkpoints[:args.num_worlds//2 - i].clone()
    checkpoints[args.num_worlds//2:args.num_worlds//2 + i] = checkpoints[:i].clone()
    sim.step()

    # Check that the observations are the same
    for j in range(len(obs)):
        print("Element", j)
        #if i == 2:
        #    continue
        #issue_elems = torch.where(~torch.isclose(obs[j][args.num_worlds//2 + i:], obs[j][:args.num_worlds//2 - i], atol=1e-2))
        #print(obs[j][args.num_worlds//2 + i:])
        #print(obs[j][:args.num_worlds//2 - i])
        assert torch.allclose(obs[j][(args.num_worlds//2 + i)*2:], obs[j][:(args.num_worlds//2 - i)*2], atol=1e-2)
        assert torch.allclose(obs[j][(args.num_worlds//2)*2:(args.num_worlds//2 + i)*2], obs[j][:i], atol=1e-2)
    # Now take an action on all worlds
    actions[..., 0] = torch.randint_like(actions[..., 0], 0, 4)
    actions[..., 1] = torch.randint_like(actions[..., 1], 0, 8)
    actions[..., 2] = torch.randint_like(actions[..., 2], 0, 5)
    actions[..., 3] = torch.randint_like(actions[..., 3], 0, 2)

    # Copy actions from first half of worlds to last half, but shifted by i
    actions[(args.num_worlds//2 + i)*2:] = actions[:(args.num_worlds//2 - i)*2].clone()
    actions[(args.num_worlds//2)*2:(args.num_worlds//2 + i)*2] = actions[:i].clone()

    sim.step()

    # Same check for observations
    for j in range(len(obs)):
        print("Element", j)
        #if i == 2:
        #    continue
        #issue_elems = torch.where(~torch.isclose(obs[j][args.num_worlds//2 + i:], obs[j][:args.num_worlds//2 - i], atol=1e-2))
        print(obs[j][(args.num_worlds//2 + i)*2:])
        print(obs[j][:(args.num_worlds//2 - i)*2])
        print(torch.where(~torch.isclose(obs[j][(args.num_worlds//2 + i)*2:], obs[j][:(args.num_worlds//2 - i)*2], atol=1e-2)))
        assert torch.allclose(obs[j][(args.num_worlds//2 + i)*2:], obs[j][:(args.num_worlds//2 - i)*2], atol=1e-2)
        assert torch.allclose(obs[j][(args.num_worlds//2)*2:(args.num_worlds//2 + i)*2], obs[j][:i], atol=1e-2)

end = time.time()

print("FPS", args.num_steps * args.num_worlds / (end - start))