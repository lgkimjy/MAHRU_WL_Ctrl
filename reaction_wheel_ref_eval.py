import argparse
import os
import re
from pathlib import Path

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


parser = argparse.ArgumentParser(description="Evaluate roll reaction wheel reference")
parser.add_argument("directory", type=str, help="Directory containing stateData.h5 file")
parser.add_argument("-i", "--initial_timestamp", type=int, default=0, help="Initial timestamp")
parser.add_argument("-e", "--end_timestamp", type=int, default=None, help="End timestamp")
parser.add_argument("--config", type=str, default="config/fsm_UnicycleCtrl_config.yaml")
parser.add_argument("--model", type=str, default="model/MAHRU-WL_w_Battery.xml")
parser.add_argument("--mass", type=float, default=None, help="Override total robot mass")
parser.add_argument("--output", type=str, default="reaction_wheel_ref_eval.png")
args = parser.parse_args()

log_file_name = "stateData.h5"
log_dir = Path(args.directory)
log_file_path = log_dir / log_file_name

assets_dir = log_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)


def _read_ds(h5, path):
    try:
        data = h5[path][:]
    except KeyError as exc:
        raise KeyError(f"Missing dataset '{path}' in {log_file_path}") from exc

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def _read_optional_ds(h5, path):
    try:
        data = h5[path][:]
    except KeyError:
        return None

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def _yaml_block(text, block_name):
    match = re.search(rf"{block_name}:\n(?P<body>(?:  .*\n)+)", text)
    return match.group("body") if match else ""


def _yaml_scalar(block, key, default):
    match = re.search(rf"^\s*{key}:\s*([-+0-9.eE]+)", block, re.M)
    return float(match.group(1)) if match else default


def _read_roll_params():
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        return 400.0, 2.0, 350.0, 1.0

    config_text = config_path.read_text()
    roll_block = _yaml_block(config_text, "roll_angular_momentum")
    kp = _yaml_scalar(roll_block, "kp", 400.0)
    kd = _yaml_scalar(roll_block, "kd", 2.0)
    max_rate = _yaml_scalar(roll_block, "max_rate", 350.0)
    sign = _yaml_scalar(roll_block, "sign", 1.0)
    return kp, kd, max_rate, sign


def _read_total_mass():
    if args.mass is not None:
        return args.mass

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        return None

    model_text = model_path.read_text()
    masses = [float(x) for x in re.findall(r'mass="([-+0-9.eE]+)"', model_text)]
    return sum(masses) if masses else None


def _rotation_matrices(flat):
    return flat.reshape(flat.shape[0], 3, 3, order="F")


def _rpy_from_rotation(R):
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


with h5py.File(log_file_path, "r") as f:
    time = _read_optional_ds(f, "time")
    hdot_ref = _read_ds(f, "ctrl/roll_momentum_rate_d")
    hdot_humanoid = _read_optional_ds(f, "ctrl/roll_momentum_rate_actual")
    p_base = _read_ds(f, "fbk/p_B")
    R_base = _read_ds(f, "fbk/R_B")
    p_com = _read_optional_ds(f, "fbk/p_CoM")
    p_contact = _read_optional_ds(f, "fbk/p_C")

lengths = [hdot_ref.shape[0], p_base.shape[0], R_base.shape[0]]
if time is not None:
    lengths.append(time.shape[0])
if hdot_humanoid is not None:
    lengths.append(hdot_humanoid.shape[0])
if p_com is not None:
    lengths.append(p_com.shape[0])
if p_contact is not None:
    lengths.append(p_contact.shape[0])
T = min(lengths)

time = np.arange(T).reshape(-1, 1) if time is None else time[:T]
hdot_ref = hdot_ref[:T]
hdot_humanoid = None if hdot_humanoid is None else hdot_humanoid[:T]
p_base = p_base[:T]
R_base = R_base[:T]
p_com = None if p_com is None else p_com[:T]
p_contact = None if p_contact is None else p_contact[:T]

start_idx = args.initial_timestamp
if args.end_timestamp is not None:
    end_idx = min(args.end_timestamp, T)
else:
    end_idx = T

time = time[start_idx:end_idx].reshape(-1)
hdot_ref = hdot_ref[start_idx:end_idx, 0]
if hdot_humanoid is not None:
    hdot_humanoid = hdot_humanoid[start_idx:end_idx, 0]
p_base = p_base[start_idx:end_idx]
R_base = R_base[start_idx:end_idx]
p_com = None if p_com is None else p_com[start_idx:end_idx]
p_contact = None if p_contact is None else p_contact[start_idx:end_idx]

kp, kd, max_rate, sign = _read_roll_params()
mass = _read_total_mass()

R = _rotation_matrices(R_base)
rpy = _rpy_from_rotation(R)

support_y = np.zeros_like(time)
if p_contact is not None and p_contact.shape[1] >= 12:
    contacts = p_contact.reshape(p_contact.shape[0], 4, 3, order="F")
    support_y = 0.5 * (contacts[:, 2, 1] + contacts[:, 3, 1])

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 1, height_ratios=[3, 2, 2])

axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

axes[0].plot(
    time,
    hdot_ref,
    linestyle="-",
    color=color_cycle[0],
    linewidth=2,
    label=r"unicycle $\dot{h}^{ref}_{roll}$",
)
if hdot_humanoid is not None:
    axes[0].plot(
        time,
        hdot_humanoid,
        linestyle="--",
        color=color_cycle[1],
        linewidth=2,
        label=r"humanoid $\dot{h}_{roll}$",
    )
else:
    axes[0].text(
        0.02,
        0.88,
        "missing ctrl/roll_momentum_rate_actual in this log",
        transform=axes[0].transAxes,
        color=color_cycle[1],
    )
axes[0].axhline(max_rate, linestyle="--", color="k", linewidth=1, alpha=0.55)
axes[0].axhline(-max_rate, linestyle="--", color="k", linewidth=1, alpha=0.55)
axes[0].set_ylabel(r"$\dot{h}_{roll}$ [Nm]")

axes[1].plot(
    time,
    p_base[:, 1] - support_y,
    linestyle="-",
    color=color_cycle[2],
    linewidth=2,
    label=r"pelvis $y-y_{support}$",
)
if p_com is not None and p_com.shape[1] >= 2:
    axes[1].plot(
        time,
        p_com[:, 1] - support_y,
        linestyle="--",
        color=color_cycle[3],
        linewidth=2,
        label=r"CoM $y-y_{support}$",
    )
axes[1].axhline(0.0, linestyle="--", color="k", linewidth=1, alpha=0.55)
axes[1].set_ylabel("support-relative y [m]")

axes[2].plot(time, rpy[:, 0], linestyle="-", color=color_cycle[4], linewidth=2, label="pelvis roll")
axes[2].plot(time, rpy[:, 1], linestyle="-", color=color_cycle[5], linewidth=2, label="pelvis pitch")
axes[2].plot(time, rpy[:, 2], linestyle="-", color=color_cycle[6], linewidth=2, label="pelvis yaw")
axes[2].axhline(0.0, linestyle="--", color="k", linewidth=1, alpha=0.55)
axes[2].set_ylabel("RPY [deg]")
axes[2].set_xlabel("time [s]")

for ax in axes:
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, ncol=3, fontsize=9, loc="best")

title = "Reaction wheel roll momentum tracking"
if mass is not None:
    title += f" | m={mass:.2f} kg"
title += f" | kp={kp:g}, kd={kd:g}, clamp=+/-{max_rate:g}"
fig.suptitle(title)

plt.tight_layout()
save_path = assets_dir / args.output
plt.savefig(save_path, dpi=300)

print(f"saved: {save_path}")
print(f"max abs unicycle hdot ref: {np.nanmax(np.abs(hdot_ref)):.6g}")
if hdot_humanoid is not None:
    print(f"max abs humanoid hdot: {np.nanmax(np.abs(hdot_humanoid)):.6g}")
print(f"max abs pelvis y support-relative: {np.nanmax(np.abs(p_base[:, 1] - support_y)):.6g}")
