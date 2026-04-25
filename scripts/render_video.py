"""Render one overcooked v2 episode from a saved checkpoint to mp4.

Plays the most trained checkpoint run_experiments.py left behind and encodes it with
the visualizer JaxMARL ships. Output is a look at the policy and nothing here writes a
result file.

Needs Pillow and an ffmpeg binary on PATH. pyproject declares neither.
"""
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
import flax.linen as nn
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

# run_experiments.py writes here too. /tmp holds until reboot, so a checkpoint from an
# earlier run has to still be on disk for the open below to find anything
OUT = Path("/tmp/exp_results")

# copied from run_experiments.py, the script that trained the checkpoint. a pickle holds a
# bare param pytree with no architecture attached, so layer count and widths and the Dense
# order have to match the writer or apply raises on a shape mismatch
# _ippo.ActorCritic carries the same four Dense layers and would load these params, with
# one extra divide by action_temperature before the logits leave the net
class AC(nn.Module):
    n_act: int; h: int = 64; nl: int = 2
    @nn.compact
    def __call__(self, x):
        z = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        for _ in range(self.nl):
            z = nn.tanh(nn.Dense(self.h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(z))
        logits = nn.Dense(self.n_act, kernel_init=nn.initializers.orthogonal(0.01))(z)
        v = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z).squeeze(-1)
        return logits, v

# the layout has to be the one the checkpoint trained on. observation on cramped_room is a
# 4x5x30 grid that flattens to 600 inputs, and another layout changes that width
env = jaxmarl.make("overcooked_v2", layout="cramped_room")
n_act = env.action_space().n
nets = {a: AC(n_act=n_act) for a in env.agents}

# last of the five run_experiments.py saves at updates 299 through 1499, so this is the
# most trained member of the partner pool
with open(OUT / "checkpoints" / "ckpt_u01499.pkl", "rb") as f:
    # the dump goes through np.asarray, so the file holds no jax types and unpickles with no
    # device attached. this map is what puts the params back on one
    params = jax.tree.map(jnp.asarray, pickle.load(f)["params"])

# 42 only picks which episode gets filmed. the reset and every action draw split off this
# one key, so a rerun produces the same video
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

states = [state]
total_ret = 0.0
# env.max_steps is 400 on this layout, so the loop stops halfway and the done break below
# never fires. 200 frames at the 8 fps set for ffmpeg is 25 seconds of video
for t in range(200):
    actions = {}
    for ag in env.agents:
        logits, _ = nets[ag].apply(params[ag], obs[ag][None])
        key, k = jax.random.split(key)
        # argmax is the other way to read this policy out. it measured 0.00 return on
        # cramped_room against 35.00 for sampling, because greedy parks on the stay action
        actions[ag] = int(jax.random.categorical(k, logits, axis=-1)[0])
    key, ks = jax.random.split(key)
    obs, state, reward, done, _ = env.step(ks, state, actions)
    states.append(state)
    # the sparse delivery channel alone. training moved on reward plus shaped_reward from
    # info, so the blend column of that run's curve reads higher than this
    # every eval in the repo sums the two agents and one soup delivery scores 20 in that sum
    total_ret += float(reward["agent_0"]) + float(reward["agent_1"])
    if bool(done["__all__"]):
        break

# states already held the reset state before the loop, so the number printed as steps is one
# above the count of env.step calls
print(f"episode: {len(states)} steps, return {total_ret:.0f}")

# render_sequence reads a time axis off each leaf of one pytree, so the python list the loop
# built is not the shape it takes
stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

viz = OvercookedV2Visualizer(tile_size=200)
frames = viz.render_sequence(stacked)
print(f"rendered {len(frames)} frames, shape {frames[0].shape}")

import numpy as np

# the frames go to disk and ffmpeg reads them back. piping them straight into the encoder
# would leave a failed run with nothing to retry against
frames_dir = OUT / "frames"
# exist_ok leaves an earlier run's PNGs where they are. ffmpeg reads frame_0000 upward and
# stops at the first missing index, so a short episode after a long one picks up the old tail
frames_dir.mkdir(exist_ok=True)
for i, f in enumerate(frames):
    arr = np.asarray(f)
    # the visualizer hands back RGBA on some versions and RGB on others. dropping alpha keeps
    # the PNGs to one format whichever version is installed
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    from PIL import Image
    Image.fromarray(arr).save(frames_dir / f"frame_{i:04d}.png")
print(f"saved {len(frames)} PNGs")

import subprocess
mp4_path = str(OUT / "eval_proper.mp4")
subprocess.run([
    # the PNGs carry no timebase, so 8 fps here is the only place playback speed is set
    "ffmpeg", "-y", "-framerate", "8",
    "-i", str(frames_dir / "frame_%04d.png"),
    # crf 18 against the libx264 default of 23 keeps the tile edges clean on flat color art.
    # preset slow costs encode time and nothing else
    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
    # yuv420p is the pixel format QuickTime opens. its chroma is subsampled by two, so
    # libx264 exits non-zero on an odd width or height
    "-pix_fmt", "yuv420p",
    # rounds both dimensions up to the next even number at a cost of one row and one column
    # of black. tile_size 200 gives even dimensions already and this covers a change to it
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    # moves the moov atom to the front of the file, so a player can start it before the whole
    # download arrives
    "-movflags", "+faststart",
    mp4_path
# check=True is what stops the print below from naming a file ffmpeg never wrote
# capture_output swallows the diagnostic and CalledProcessError shows the exit status alone.
# read e.stderr to learn why an encode failed
], check=True, capture_output=True)
print(f"saved {mp4_path}")
