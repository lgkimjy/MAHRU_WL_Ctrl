import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluate p_CoM data')
parser.add_argument('directory', type=str, help='Directory containing stateData.h5 file')
parser.add_argument('-i', '--initial_timestamp', type=int, default=0, help='Initial timestamp') # add -i option to the command
parser.add_argument('-e', '--end_timestamp', type=int, default=None, help='End timestamp') # add -e option to the command
args = parser.parse_args()

log_file_name = "stateData.h5"
log_dir = Path(args.directory)
log_file_path = log_dir / log_file_name

# Create assets directory if it doesn't exist
assets_dir = log_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(log_file_path, 'r') as f:
    fbk = f['fbk/p_CoM'][:]
    cmd = f['ctrl/p_CoM_d'][:]

# cmd = np.genfromtxt("./p_CoM_d.txt", delimiter=",")
# fbk = np.genfromtxt("./p_CoM.txt", delimiter=",")

if cmd.ndim == 1:
    cmd = cmd.reshape(-1, 1)
if fbk.ndim == 1:
    fbk = fbk.reshape(-1, 1)

# ---- time alignment ----
start_idx = args.initial_timestamp
if args.end_timestamp is not None:
    end_idx = min(args.end_timestamp, cmd.shape[0])
else:
    end_idx = cmd.shape[0]
cmd = cmd[start_idx:end_idx]
fbk = fbk[start_idx:end_idx]
t = np.arange(start_idx, end_idx)

axis_names = [
    'x-axis',
    'y-axis',
    'z-axis'
]

height_ratios = [3, 3, 3]

# ---- figure & gridspec ----
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, height_ratios=height_ratios)

axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

# color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ---- plotting ----
for i in range(3):
    axes[i].plot(t, fbk[:, i],
            linestyle='--',
            color=color_cycle[i],
            label=f'fbk {axis_names[i]}')

    axes[i].plot(t, cmd[:, i],
            linestyle='-',
            color=color_cycle[i],
            linewidth=2,
            alpha=0.5,
            label=f'cmd {axis_names[i]}')
    axes[i].grid(True)

# ---- labels ----
for i in range(3):
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel(f'p_CoM')

# ---- legend ----
for i in range(3):
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles, labels, ncol=3, fontsize=9, loc='best')

plt.tight_layout()
save_path = assets_dir / "p_CoM_eval.png"
plt.savefig(save_path, dpi=300)
plt.show()
