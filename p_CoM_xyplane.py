import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import h5py
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="CoM / ZMP trajectories in the xy plane")
parser.add_argument("directory", type=str, help="Directory containing stateData.h5 file")
parser.add_argument("-i", "--initial_timestamp", type=int, default=0, help="Initial timestamp")
parser.add_argument("-e", "--end_timestamp", type=int, default=None, help="End timestamp")
args = parser.parse_args()

log_file_name = "stateData.h5"
log_dir = Path(args.directory)
log_file_path = log_dir / log_file_name

assets_dir = log_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)


def _read_ds(h5, path):
    try:
        return h5[path][:]
    except KeyError:
        return None


def _as_2cols(a):
    if a is None:
        return None
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a[:, :2]


# Colors match PhaseType in include/ContactFSM.hpp (enum order from 0)
phase_colors = {
    0: "#CA534B",   # SSP_LEFT
    1: "#4B6CDA",   # SSP_RIGHT
}
_ds = "#888888"
for k in range(2, 12):
    phase_colors[k] = _ds

phase_labels = {
    0: "SSP_LEFT",
    1: "SSP_RIGHT",
    2: "DSP_LFRF",
}


def phase_int(ph):
    return int(np.asarray(ph).reshape(-1)[0])


def get_phase_color(ph_i):
    return phase_colors.get(ph_i, "#bbbbbb")


def iter_phase_segments(phase_arr):
    n = len(phase_arr)
    seg_start = 0
    while seg_start < n:
        curr = phase_int(phase_arr[seg_start])
        seg_end = seg_start + 1
        while seg_end < n and phase_int(phase_arr[seg_end]) == curr:
            seg_end += 1
        yield seg_start, seg_end, curr
        seg_start = seg_end


def plot_xy_by_phase(ax, xy, phase, linestyle, linewidth, label):
    if phase is None or len(phase) != len(xy):
        ax.plot(xy[:, 0], xy[:, 1], linestyle=linestyle, linewidth=linewidth, label=label)
        return
    first = True
    for seg_start, seg_end, ph_i in iter_phase_segments(phase):
        sl = slice(seg_start, seg_end)
        ax.plot(
            xy[sl, 0],
            xy[sl, 1],
            linestyle=linestyle,
            linewidth=linewidth,
            color=get_phase_color(ph_i),
            label=label if first else None,
        )
        first = False


with h5py.File(log_file_path, "r") as f:
    fbk = f["fbk/p_CoM"][:]
    cmd = f["ctrl/p_CoM_d"][:]
    zmp = f["fbk/p_ZMP"][:]
    p_ZMP_ref = _read_ds(f, "ctrl/p_ZMP_ref")
    p_ZMP_d = _read_ds(f, "ctrl/p_ZMP_d")
    phase = _read_ds(f, "ctrl/contact_phase")

if cmd.ndim == 1:
    cmd = cmd.reshape(-1, 1)
if fbk.ndim == 1:
    fbk = fbk.reshape(-1, 1)
if zmp.ndim == 1:
    zmp = zmp.reshape(-1, 1)

lengths = [cmd.shape[0], fbk.shape[0], zmp.shape[0]]
if p_ZMP_ref is not None:
    lengths.append(p_ZMP_ref.shape[0])
if p_ZMP_d is not None:
    lengths.append(p_ZMP_d.shape[0])
if phase is not None:
    lengths.append(np.asarray(phase).reshape(-1).shape[0])
T = min(lengths)

fbk = fbk[:T]
cmd = cmd[:T]
zmp = zmp[:T]
if p_ZMP_ref is not None:
    p_ZMP_ref = p_ZMP_ref[:T]
if p_ZMP_d is not None:
    p_ZMP_d = p_ZMP_d[:T]
if phase is not None:
    phase = np.asarray(phase).reshape(-1)[:T].astype(np.int64, copy=False)

start_idx = args.initial_timestamp
if args.end_timestamp is not None:
    end_idx = min(args.end_timestamp, T)
else:
    end_idx = T

cmd = cmd[start_idx:end_idx]
fbk = fbk[start_idx:end_idx]
zmp = zmp[start_idx:end_idx]
if p_ZMP_ref is not None:
    p_ZMP_ref = p_ZMP_ref[start_idx:end_idx]
if p_ZMP_d is not None:
    p_ZMP_d = p_ZMP_d[start_idx:end_idx]
if phase is not None:
    phase = phase[start_idx:end_idx]

fbk_xy = _as_2cols(fbk)
cmd_xy = _as_2cols(cmd)
zmp_xy = _as_2cols(zmp)
ref_xy = _as_2cols(p_ZMP_ref)
des_xy = _as_2cols(p_ZMP_d)

fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1])
ax_com = fig.add_subplot(gs[0, 0])
ax_zmp = fig.add_subplot(gs[1, 0])

plot_xy_by_phase(ax_com, cmd_xy, phase, linestyle="-", linewidth=2, label="p_CoM_d")
plot_xy_by_phase(ax_com, fbk_xy, phase, linestyle="--", linewidth=3, label="p_CoM")
ax_com.set_xlabel("x [m]")
ax_com.set_ylabel("y [m]")
ax_com.set_title("Center of Mass")
ax_com.set_aspect("equal", adjustable="datalim")
ax_com.grid(True, alpha=0.35)

plot_xy_by_phase(ax_zmp, zmp_xy, phase, linestyle="-", linewidth=2, label="p_ZMP")
if ref_xy is not None:
    plot_xy_by_phase(ax_zmp, ref_xy, phase, linestyle="-", linewidth=2, label="p_ZMP_ref")
if des_xy is not None:
    plot_xy_by_phase(ax_zmp, des_xy, phase, linestyle="--", linewidth=2, label="p_ZMP_d")
ax_zmp.set_xlabel("x [m]")
ax_zmp.set_ylabel("y [m]")
ax_zmp.set_title("Zero Moment Point")
ax_zmp.set_aspect("equal", adjustable="datalim")
ax_zmp.grid(True, alpha=0.35)

span_alpha = 0.28
if phase is not None and len(phase) > 0:
    for ax in (ax_com, ax_zmp):
        line_handles, line_labels = ax.get_legend_handles_labels()
        phase_handles = [
            Patch(
                facecolor=get_phase_color(int(ph_i)),
                alpha=span_alpha,
                edgecolor="none",
                label=phase_labels.get(int(ph_i), f"phase {int(ph_i)}"),
            )
            for ph_i in sorted(np.unique(phase))
        ]
        ax.legend(handles=list(line_handles) + phase_handles, loc="best")
else:
    ax_com.legend(loc="best")
    ax_zmp.legend(loc="best")

plt.tight_layout()
save_path = assets_dir / "p_CoM_xyplane.png"
plt.savefig(save_path, dpi=300)
plt.show()
