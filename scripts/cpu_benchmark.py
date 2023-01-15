import madrona_python
import gpu_hideseek_python
import torch
import sys
import time
import PIL
import PIL.Image

num_worlds = int(sys.argv[1])
num_steps = int(sys.argv[2])
entities_per_world = int(sys.argv[3])
reset_chance = float(sys.argv[4])

render_width = 30 
render_height = 1 

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CPU,
        gpu_id = 0,
        num_worlds = num_worlds,
        min_entities_per_world = entities_per_world,
        max_entities_per_world = entities_per_world,
        render_width = render_width,
        render_height = render_height,
        debug_compile = False,
)

#actions = sim.action_tensor().to_torch()
#resets = sim.reset_tensor().to_torch()
#rgb_observations = sim.rgb_tensor().to_torch()
#print(actions.shape, actions.dtype)
#print(resets.shape, resets.dtype)
#print(rgb_observations.shape, rgb_observations.dtype)

start = time.time()

#reset_no = torch.zeros_like(resets[:, 0], dtype=torch.int32)
#reset_yes = torch.ones_like(resets[:, 0], dtype=torch.int32)
#reset_rand = torch.zeros_like(resets[:, 0], dtype=torch.float32)
#
#resets[:, 0] = 1
#resets[:, 1] = 3
#resets[:, 2] = 2

def dump_obs(dump_dir, step_idx):
    N = rgb_observations.shape[0]
    A = rgb_observations.shape[1]

    num_wide = min(64, N * A)

    reshaped = rgb_observations.reshape(N * A // num_wide, num_wide, *rgb_observations.shape[2:])
    grid = reshaped.permute(0, 2, 1, 3, 4)

    grid = grid.reshape(N * A // num_wide * render_height, num_wide * render_width, 4)
    grid = grid.type(torch.uint8).cpu().numpy()

    img = PIL.Image.fromarray(grid)
    img.save(f"{dump_dir}/{step_idx}.png", format="PNG")

print("hi")
sim.step()

for i in range(num_steps):
    sim.step()

    #torch.rand(reset_rand.shape, out=reset_rand)

    #reset_cond = torch.where(reset_rand < reset_chance, reset_yes, reset_no)
    #resets[:, 0].copy_(reset_cond)

    #actions[..., 0:2] = torch.randint_like(actions[..., 0:2], -5, 5)

    if len(sys.argv) > 5:
        dump_obs(sys.argv[5], i)

end = time.time()

duration = end - start
print(num_worlds * num_steps / duration, duration)

del sim