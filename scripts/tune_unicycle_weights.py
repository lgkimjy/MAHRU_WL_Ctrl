#!/usr/bin/env python
import argparse
import csv
import itertools
import math
import random
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTED_CONFIG = ROOT / "config" / "config_weighted.yaml"
FSM_CONFIG = ROOT / "config" / "fsm_UnicycleCtrl_config.yaml"


def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.4g}"
    return f"{value:.6g}"


def replace_top_scalar(text, key, value):
    pattern = rf"(?m)^{re.escape(key)}:\s*(?:[-+0-9.eE]+|true|false)"
    replacement = f"{key}: {fmt(value)}"
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise KeyError(f"missing scalar key {key}")
    return new_text


def replace_block_scalar(text, block_name, key, value):
    block_pattern = rf"(?m)^{re.escape(block_name)}:\n(?:  .*\n)+"
    match = re.search(block_pattern, text)
    if not match:
        raise KeyError(f"missing block {block_name}")

    block = match.group(0)
    scalar_pattern = rf"(?m)^  {re.escape(key)}:\s*(?:[-+0-9.eE]+|true|false)"
    new_block, count = re.subn(
        scalar_pattern, f"  {key}: {fmt(value)}", block, count=1
    )
    if count != 1:
        raise KeyError(f"missing scalar key {block_name}.{key}")
    return text[: match.start()] + new_block + text[match.end() :]


def replace_nested_block_scalar(text, parent_name, block_name, key, value):
    parent_pattern = rf"(?m)^{re.escape(parent_name)}:\n(?:  .*\n)+"
    parent_match = re.search(parent_pattern, text)
    if not parent_match:
        raise KeyError(f"missing block {parent_name}")

    parent = parent_match.group(0)
    block_pattern = rf"(?m)^  {re.escape(block_name)}:\n(?:    .*\n)+"
    block_match = re.search(block_pattern, parent)
    if not block_match:
        raise KeyError(f"missing block {parent_name}.{block_name}")

    block = block_match.group(0)
    scalar_pattern = rf"(?m)^    {re.escape(key)}:\s*(?:[-+0-9.eE]+|true|false)"
    new_block, count = re.subn(
        scalar_pattern, f"    {key}: {fmt(value)}", block, count=1
    )
    if count != 1:
        raise KeyError(f"missing scalar key {parent_name}.{block_name}.{key}")

    new_parent = (
        parent[: block_match.start()]
        + new_block
        + parent[block_match.end() :]
    )
    return text[: parent_match.start()] + new_parent + text[parent_match.end() :]


def replace_indented_vector(text, key, values):
    values_text = ", ".join(fmt(v) for v in values)
    pattern = rf"(?m)^  {re.escape(key)}:\s*\[[^\]]+\]"
    new_text, count = re.subn(pattern, f"  {key}: [{values_text}]", text, count=1)
    if count != 1:
        raise KeyError(f"missing vector key {key}")
    return new_text


def apply_candidate(weighted_text, fsm_text, candidate):
    for key in (
        "W_Centroidal",
        "W_swingLeg",
        "W_JointAcc",
        "W_CentroidalForce",
        "W_RollAngularMomentum",
        "W_SwingLegRollMomentum",
        "W_SwingLateralAccel",
        "W_TorsoYawJointAcc",
        "W_wheelAccel",
    ):
        if key in candidate:
            weighted_text = replace_top_scalar(weighted_text, key, candidate[key])

    if "orientation_kp_xy" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "orientation_kp", [candidate["orientation_kp_xy"], candidate["orientation_kp_xy"], 10]
        )
    if "orientation_kp_roll" in candidate or "orientation_kp_pitch" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text,
            "orientation_kp",
            [
                candidate.get("orientation_kp_roll", 50),
                candidate.get("orientation_kp_pitch", 200),
                10,
            ],
        )
    if "orientation_kd_xy" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "orientation_kd", [candidate["orientation_kd_xy"], candidate["orientation_kd_xy"], 1]
        )
    if "orientation_kd_roll" in candidate or "orientation_kd_pitch" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text,
            "orientation_kd",
            [
                candidate.get("orientation_kd_roll", 10),
                candidate.get("orientation_kd_pitch", 20),
                1,
            ],
        )
    if "com_kp_xy" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "com_kp", [candidate["com_kp_xy"], candidate["com_kp_xy"], 500]
        )
    if "com_kd_xy" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "com_kd", [candidate["com_kd_xy"], candidate["com_kd_xy"], 15]
        )
    if "swing_kp_y" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "kp", [1000, candidate["swing_kp_y"], 1000]
        )
    if "swing_kd_y" in candidate:
        weighted_text = replace_indented_vector(
            weighted_text, "kd", [40, candidate["swing_kd_y"], 40]
        )

    for key, yaml_key in (
        ("com_shift_duration", "duration"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "com_shift", yaml_key, candidate[key]
            )
    if "com_shift_y_offset" in candidate:
        fsm_text = replace_indented_vector(
            fsm_text, "offset_m", [0.0, candidate["com_shift_y_offset"], 0.0]
        )

    for key, yaml_key in (
        ("roll_kp", "kp"),
        ("roll_kd", "kd"),
        ("roll_angle_kp", "roll_angle_kp"),
        ("roll_rate_kd", "roll_rate_kd"),
        ("roll_max_rate", "max_rate"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "roll_angular_momentum", yaml_key, candidate[key]
            )
    if "swing_leg_roll_enabled" in candidate:
        fsm_text = replace_block_scalar(
            fsm_text,
            "swing_leg_roll_momentum",
            "enabled",
            candidate["swing_leg_roll_enabled"],
        )
    for key, yaml_key in (
        ("swing_leg_roll_scale", "scale"),
        ("swing_leg_roll_max_rate", "max_rate"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "swing_leg_roll_momentum", yaml_key, candidate[key]
            )
    if "lateral_contact_kd" in candidate:
        fsm_text = replace_block_scalar(
            fsm_text, "single_wheel_stance", "lateral_contact_kd", candidate["lateral_contact_kd"]
        )
    for key, yaml_key in (
        ("pitch_kp", "kp"),
        ("pitch_kd", "kd"),
        ("pitch_accel_kp", "accel_kp"),
        ("pitch_accel_kd", "accel_kd"),
        ("pitch_sign", "sign"),
        ("pitch_max_lin_vel", "max_lin_vel"),
        ("pitch_max_lin_acc", "max_lin_acc"),
    ):
        if key in candidate:
            fsm_text = replace_nested_block_scalar(
                fsm_text, "single_wheel_stance", "pitch_control", yaml_key, candidate[key]
            )
    for key, yaml_key in (
        ("lift_height", "lift_height"),
        ("lift_duration", "lift_duration"),
        ("swing_clearance_min", "min_clearance"),
        ("swing_clearance_kp", "clearance_kp"),
        ("swing_clearance_kd", "clearance_kd"),
        ("swing_clearance_max_acc", "clearance_max_acc"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "right_foot_lift", yaml_key, candidate[key]
            )
    if "swing_reaction_enabled" in candidate:
        fsm_text = replace_block_scalar(
            fsm_text,
            "swing_leg_reaction",
            "enabled",
            candidate["swing_reaction_enabled"],
        )
    for key, yaml_key in (
        ("swing_reaction_sign", "sign"),
        ("swing_reaction_roll_kp", "roll_kp"),
        ("swing_reaction_roll_rate_kd", "roll_rate_kd"),
        ("swing_reaction_com_sign", "com_sign"),
        ("swing_reaction_com_kp", "com_kp"),
        ("swing_reaction_com_kd", "com_kd"),
        ("swing_reaction_max_offset", "max_offset"),
        ("swing_reaction_max_vel", "max_vel"),
        ("swing_reaction_tau", "tau"),
        ("swing_accel_enabled", "accel_task_enabled"),
        ("swing_accel_sign", "accel_sign"),
        ("swing_accel_roll_kp", "accel_roll_kp"),
        ("swing_accel_roll_rate_kd", "accel_roll_rate_kd"),
        ("swing_accel_momentum_scale", "accel_momentum_scale"),
        ("swing_accel_max", "accel_max"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "swing_leg_reaction", yaml_key, candidate[key]
            )

    for key, yaml_key in (
        ("line_roll_com_enabled", "roll_com_feedback_enabled"),
        ("line_roll_com_sign", "roll_com_sign"),
        ("line_roll_com_kp", "roll_com_kp"),
        ("line_roll_com_kd", "roll_com_kd"),
        ("line_roll_com_max_offset", "roll_com_max_offset"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "line_contact_balance", yaml_key, candidate[key]
            )
    for key, yaml_key in (
        ("line_moment_x_mask", "moment_x_mask"),
        ("line_moment_y_mask", "moment_y_mask"),
    ):
        if key in candidate:
            fsm_text = replace_block_scalar(
                fsm_text, "line_contact_balance", yaml_key, candidate[key]
            )

    for key, yaml_key in (
        ("com_feedback_kp", "kp"),
        ("com_feedback_kd", "kd"),
        ("com_feedback_sign", "sign"),
        ("com_feedback_max_offset", "max_offset"),
    ):
        if key in candidate:
            fsm_text = replace_nested_block_scalar(
                fsm_text, "single_wheel_stance", "com_feedback", yaml_key, candidate[key]
            )

    return weighted_text, fsm_text


def rotation_matrices(flat):
    return flat.reshape(flat.shape[0], 3, 3, order="F")


def rpy_from_rotation(flat):
    R = rotation_matrices(flat)
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


def rms(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(np.square(x))))


def read_ds(h5, path, default=None):
    if path not in h5:
        return default
    data = h5[path][:]
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def evaluate_log(log_dir, requested_duration):
    h5_path = log_dir / "stateData.h5"
    if not h5_path.exists():
        return {"ok": 0, "score": 1e9, "failure": "missing_h5"}

    try:
        with h5py.File(h5_path, "r") as h5:
            time = read_ds(h5, "time")
            p_base = read_ds(h5, "fbk/p_B")
            p_com = read_ds(h5, "fbk/p_CoM")
            r_base = read_ds(h5, "fbk/R_B")
            p_contact = read_ds(h5, "fbk/p_C")
            hdot_ref = read_ds(h5, "ctrl/roll_momentum_rate_d")
            hdot_actual = read_ds(h5, "ctrl/roll_momentum_rate_actual")
            hdot_wbc = read_ds(h5, "ctrl/roll_momentum_rate_wbc")
            lift_phase = read_ds(h5, "ctrl/right_foot_lift_phase")
            single_phase = read_ds(h5, "ctrl/single_wheel_phase")
            torque = read_ds(h5, "ctrl/torq_d")
    except OSError as exc:
        return {"ok": 0, "score": 1e9, "failure": f"h5_error:{exc}"}

    arrays = [time, p_base, p_com, r_base, p_contact, hdot_ref, hdot_actual, hdot_wbc, lift_phase]
    n = min(a.shape[0] for a in arrays if a is not None)
    if n < 10:
        return {"ok": 0, "score": 1e9, "failure": "short_log", "samples": n}

    time = time[:n, 0]
    p_base = p_base[:n]
    p_com = p_com[:n]
    rpy = rpy_from_rotation(r_base[:n])
    contacts = p_contact[:n].reshape(n, 4, 3, order="F")
    line_support_y = 0.5 * (contacts[:, 2, 1] + contacts[:, 3, 1])
    wheel_support_y = contacts[:, 3, 1]
    if single_phase is not None:
        phase = single_phase[:n, 0]
        support_y = np.where(phase > 0.5, wheel_support_y, line_support_y)
    else:
        support_y = line_support_y

    hdot_ref = hdot_ref[:n, 0]
    hdot_actual = hdot_actual[:n, 0]
    hdot_wbc = hdot_wbc[:n, 0]
    lift_phase = lift_phase[:n, 0]
    torque = None if torque is None else torque[:n]

    active = lift_phase > 0.05
    if np.count_nonzero(active) < 20:
        active = time > max(0.5 * requested_duration, time[-1] - 1.0)
    if np.count_nonzero(active) < 1:
        active = np.ones_like(time, dtype=bool)

    clearance_active = active.copy()
    active_indices = np.flatnonzero(active)
    if active_indices.size:
        clearance_active &= time >= time[active_indices[0]] + 0.45
    if np.count_nonzero(clearance_active) < 1:
        clearance_active = active

    com_y = p_com[:, 1] - support_y
    pelvis_y = p_base[:, 1] - support_y
    right_swing_min_z = float(np.nanmin(contacts[clearance_active, :2, 2]))
    clearance_target = 0.035
    clearance_violation = max(0.0, clearance_target - right_swing_min_z)
    active_hdot_abs = np.abs(hdot_ref[active])
    hdot_scale = max(100.0, float(np.nanpercentile(active_hdot_abs, 95)))
    hdot_ref_max = float(np.nanmax(active_hdot_abs)) if active_hdot_abs.size else 0.0

    min_com_z = float(np.nanmin(p_com[:, 2]))
    max_roll_abs = float(np.nanmax(np.abs(rpy[:, 0])))
    max_pitch_abs = float(np.nanmax(np.abs(rpy[:, 1])))
    survived = float(time[-1])
    early_stop = survived < requested_duration - 0.02
    fall = early_stop or min_com_z < 0.45 or max_roll_abs > 55.0 or max_pitch_abs > 55.0

    metrics = {
        "ok": 1,
        "score": 0.0,
        "failure": "",
        "samples": n,
        "survived_s": survived,
        "rms_com_y_m": rms(com_y[active]),
        "rms_pelvis_y_m": rms(pelvis_y[active]),
        "rms_roll_deg": rms(rpy[active, 0]),
        "rms_pitch_deg": rms(rpy[active, 1]),
        "rms_yaw_deg": rms(rpy[active, 2]),
        "rms_hdot_actual_err_norm": rms((hdot_ref[active] - hdot_actual[active]) / hdot_scale),
        "rms_hdot_wbc_err_norm": rms((hdot_ref[active] - hdot_wbc[active]) / hdot_scale),
        "sat_frac": float(np.mean(np.abs(hdot_ref[active]) > 0.98 * hdot_ref_max))
        if hdot_ref_max > 0.0 else 0.0,
        "min_com_z_m": min_com_z,
        "max_roll_abs_deg": max_roll_abs,
        "max_pitch_abs_deg": max_pitch_abs,
        "right_swing_min_z_m": right_swing_min_z,
        "clearance_violation_m": clearance_violation,
        "early_stop": int(early_stop),
        "fall": int(fall),
    }
    metrics["rms_tau"] = rms(torque[active]) if torque is not None else float("nan")

    survival_penalty = 25.0 * max(0.0, requested_duration - survived)
    fall_penalty = 100.0 if fall else 0.0
    metrics["score"] = (
        140.0 * metrics["rms_com_y_m"]
        + 25.0 * metrics["rms_pelvis_y_m"]
        + 0.45 * metrics["rms_roll_deg"]
        + 0.20 * metrics["rms_pitch_deg"]
        + 0.04 * metrics["rms_yaw_deg"]
        + 7.0 * metrics["rms_hdot_actual_err_norm"]
        + 4.0 * metrics["rms_hdot_wbc_err_norm"]
        + 1.5 * metrics["sat_frac"]
        + 900.0 * clearance_violation
        + survival_penalty
        + fall_penalty
    )
    return metrics


def text_from_timeout_stream(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def candidate_stream(trials, seed):
    baseline = {"name": "baseline"}
    yield baseline

    rng = random.Random(seed)
    grids = {
        "W_Centroidal": [5, 10, 25, 50],
        "W_swingLeg": [50, 100, 200, 400],
        "W_JointAcc": [10, 20, 60, 120],
        "W_CentroidalForce": [50, 100, 200, 400],
        "W_RollAngularMomentum": [500, 1000, 1500, 2500, 4000],
        "W_SwingLegRollMomentum": [0, 500, 1000, 3000, 7000],
        "W_SwingLateralAccel": [0, 1000, 3000, 10000, 20000],
        "W_TorsoYawJointAcc": [500, 1000, 3000, 7000],
        "W_wheelAccel": [20, 50, 100, 200],
        "roll_kp": [150, 250, 400, 650, 900],
        "roll_kd": [10, 20, 35, 60],
        "roll_angle_kp": [0.0, 1.0, 2.0, 4.0, 8.0],
        "roll_rate_kd": [0.0, 0.5, 1.0, 2.0, 4.0],
        "roll_max_rate": [500, 800, 1000, 1400],
        "lateral_contact_kd": [10, 20, 50, 100],
        "pitch_kp": [100, 250, 500, 900, 1400],
        "pitch_kd": [2, 5, 10, 20],
        "pitch_accel_kp": [0.0, 4.0, 8.0, 12.0],
        "pitch_accel_kd": [0.0, 0.5, 1.0, 2.0],
        "pitch_sign": [-1.0, 1.0],
        "pitch_max_lin_vel": [0.15, 0.25, 0.4, 0.6],
        "pitch_max_lin_acc": [0.0, 2.0, 4.0, 6.0],
        "orientation_kp_xy": [50, 100, 200],
        "orientation_kd_xy": [6, 10, 18],
        "orientation_kp_roll": [30, 50, 80],
        "orientation_kp_pitch": [150, 200, 250],
        "orientation_kd_roll": [6, 10, 14],
        "orientation_kd_pitch": [15, 20, 28],
        "com_kp_xy": [70, 100, 160],
        "com_kd_xy": [2, 5, 10],
        "com_shift_y_offset": [-0.04, -0.02, 0.0, 0.02],
        "com_shift_duration": [2.0, 3.0, 4.0],
        "swing_kp_y": [20, 50, 100, 300, 1000],
        "swing_kd_y": [2, 5, 10, 20, 40],
        "swing_leg_roll_enabled": [False, True],
        "swing_leg_roll_scale": [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0],
        "swing_leg_roll_max_rate": [100, 300, 600],
        "lift_height": [0.10, 0.14, 0.18],
        "lift_duration": [0.3, 0.6, 1.0],
        "swing_clearance_min": [0.035, 0.05, 0.07],
        "swing_clearance_kp": [400, 800, 1200],
        "swing_clearance_kd": [40, 60, 90],
        "swing_clearance_max_acc": [80, 120],
        "swing_reaction_enabled": [False, True],
        "swing_reaction_sign": [-1.0, 1.0],
        "swing_reaction_roll_kp": [0.10, 0.20, 0.35, 0.50],
        "swing_reaction_roll_rate_kd": [0.00, 0.03, 0.06, 0.10],
        "swing_reaction_com_sign": [1.0],
        "swing_reaction_com_kp": [0.20, 0.35, 0.50, 0.70],
        "swing_reaction_com_kd": [0.06, 0.12, 0.18, 0.24],
        "swing_reaction_max_offset": [0.08, 0.10, 0.12, 0.14],
        "swing_reaction_max_vel": [0.8, 1.0, 1.2, 1.6],
        "swing_reaction_tau": [0.05, 0.10, 0.16],
        "swing_accel_enabled": [False, True],
        "swing_accel_sign": [-1.0, 1.0],
        "swing_accel_roll_kp": [0.0, 20.0, 50.0, 100.0],
        "swing_accel_roll_rate_kd": [0.0, 5.0, 10.0, 20.0],
        "swing_accel_momentum_scale": [-0.8, -0.35, 0.0, 0.35, 0.8],
        "swing_accel_max": [10.0, 25.0, 50.0, 80.0],
        "line_roll_com_enabled": [False, True],
        "line_roll_com_sign": [-1.0, 1.0],
        "line_roll_com_kp": [0.0, 0.04, 0.08, 0.12, 0.20],
        "line_roll_com_kd": [0.0, 0.005, 0.01, 0.02],
        "line_roll_com_max_offset": [0.0, 0.006, 0.012, 0.02, 0.035],
        "line_moment_x_mask": [0.6, 0.8, 1.0],
        "line_moment_y_mask": [0.8, 1.0],
        "com_feedback_kp": [0.03, 0.04, 0.05],
        "com_feedback_kd": [0.005, 0.01, 0.015],
        "com_feedback_max_offset": [0.05, 0.06, 0.08],
    }

    handpicked = [
        {
            "name": "best25_pitch_accel",
            "W_Centroidal": 5,
            "W_swingLeg": 100,
            "W_JointAcc": 120,
            "W_CentroidalForce": 200,
            "W_RollAngularMomentum": 2500,
            "W_SwingLegRollMomentum": 500,
            "W_TorsoYawJointAcc": 1000,
            "W_wheelAccel": 200,
            "roll_kp": 250,
            "roll_kd": 35,
            "roll_angle_kp": 2.0,
            "roll_rate_kd": 1.0,
            "roll_max_rate": 1000,
            "lateral_contact_kd": 50,
            "pitch_kp": 900,
            "pitch_kd": 20,
            "pitch_accel_kp": 8.0,
            "pitch_accel_kd": 1.0,
            "pitch_sign": 1.0,
            "pitch_max_lin_vel": 0.4,
            "pitch_max_lin_acc": 4.0,
            "orientation_kp_xy": 50,
            "orientation_kd_xy": 10,
            "com_kp_xy": 100,
            "com_kd_xy": 5,
            "com_shift_y_offset": 0.0,
            "com_shift_duration": 4.0,
            "swing_kp_y": 20,
            "swing_kd_y": 5,
            "swing_leg_roll_enabled": False,
            "swing_leg_roll_scale": 0.25,
            "swing_leg_roll_max_rate": 600,
            "swing_reaction_enabled": True,
            "swing_reaction_sign": 1.0,
            "swing_reaction_roll_kp": 0.2,
            "swing_reaction_roll_rate_kd": 0.1,
            "swing_reaction_max_offset": 0.08,
            "swing_reaction_max_vel": 0.8,
            "swing_reaction_tau": 0.16,
            "lift_height": 0.14,
            "lift_duration": 1.0,
            "swing_clearance_min": 0.07,
            "swing_clearance_kp": 400,
            "swing_clearance_kd": 40,
            "swing_clearance_max_acc": 80,
        },
        {
            "name": "best146_more_roll",
            "W_Centroidal": 50,
            "W_swingLeg": 50,
            "W_JointAcc": 20,
            "W_CentroidalForce": 400,
            "W_RollAngularMomentum": 2500,
            "W_SwingLegRollMomentum": 0,
            "W_TorsoYawJointAcc": 1000,
            "W_wheelAccel": 250,
            "roll_kp": 250,
            "roll_kd": 20,
            "roll_angle_kp": 3.0,
            "roll_rate_kd": 2.0,
            "roll_max_rate": 1000,
            "lateral_contact_kd": 50,
            "pitch_kp": 500,
            "pitch_kd": 20,
            "pitch_accel_kp": 6.0,
            "pitch_accel_kd": 1.0,
            "pitch_sign": -1.0,
            "pitch_max_lin_vel": 0.25,
            "pitch_max_lin_acc": 4.0,
            "orientation_kp_xy": 50,
            "orientation_kd_xy": 10,
            "com_kp_xy": 100,
            "com_kd_xy": 5,
            "com_shift_y_offset": 0.0,
            "com_shift_duration": 4.0,
            "swing_kp_y": 100,
            "swing_kd_y": 5,
            "swing_leg_roll_enabled": False,
            "swing_leg_roll_scale": -1.0,
            "swing_leg_roll_max_rate": 100,
            "swing_reaction_enabled": False,
            "swing_reaction_sign": 1.0,
            "swing_reaction_roll_kp": 0.35,
            "swing_reaction_roll_rate_kd": 0.1,
            "swing_reaction_max_offset": 0.08,
            "swing_reaction_max_vel": 0.8,
            "swing_reaction_tau": 0.05,
            "lift_height": 0.1,
            "lift_duration": 0.6,
            "swing_clearance_min": 0.035,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
        },
        {
            "name": "swing_soft",
            "W_Centroidal": 25,
            "W_swingLeg": 50,
            "W_JointAcc": 20,
            "W_CentroidalForce": 50,
            "W_RollAngularMomentum": 4000,
            "W_SwingLegRollMomentum": 3000,
            "W_TorsoYawJointAcc": 3000,
            "W_wheelAccel": 20,
            "roll_kp": 400,
            "roll_kd": 35,
            "roll_angle_kp": 2.0,
            "roll_rate_kd": 1.0,
            "roll_max_rate": 800,
            "lateral_contact_kd": 50,
            "pitch_accel_kp": 6.0,
            "pitch_accel_kd": 1.0,
            "pitch_max_lin_acc": 4.0,
            "orientation_kp_xy": 50,
            "orientation_kd_xy": 18,
            "com_kp_xy": 160,
            "com_kd_xy": 10,
            "swing_kp_y": 100,
            "swing_kd_y": 10,
            "swing_leg_roll_enabled": True,
            "swing_leg_roll_scale": -0.5,
            "swing_leg_roll_max_rate": 300,
            "swing_reaction_enabled": True,
            "swing_reaction_sign": 1.0,
            "swing_reaction_roll_kp": 0.20,
            "swing_reaction_roll_rate_kd": 0.03,
            "swing_reaction_max_offset": 0.08,
            "swing_reaction_max_vel": 0.6,
            "swing_reaction_tau": 0.12,
            "lift_height": 0.14,
            "swing_clearance_min": 0.05,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
            "lift_duration": 0.6,
            "com_shift_y_offset": 0.0,
            "com_shift_duration": 3.0,
        },
        {
            "name": "swing_rate_only",
            "W_Centroidal": 25,
            "W_swingLeg": 50,
            "W_JointAcc": 20,
            "W_CentroidalForce": 50,
            "W_RollAngularMomentum": 4000,
            "W_SwingLegRollMomentum": 3000,
            "W_TorsoYawJointAcc": 3000,
            "W_wheelAccel": 20,
            "roll_kp": 400,
            "roll_kd": 35,
            "roll_angle_kp": 4.0,
            "roll_rate_kd": 1.0,
            "roll_max_rate": 800,
            "lateral_contact_kd": 50,
            "pitch_accel_kp": 8.0,
            "pitch_accel_kd": 1.0,
            "pitch_max_lin_acc": 4.0,
            "orientation_kp_xy": 50,
            "orientation_kd_xy": 18,
            "com_kp_xy": 160,
            "com_kd_xy": 10,
            "swing_kp_y": 100,
            "swing_kd_y": 10,
            "swing_leg_roll_enabled": True,
            "swing_leg_roll_scale": -0.5,
            "swing_leg_roll_max_rate": 300,
            "swing_reaction_enabled": True,
            "swing_reaction_sign": 1.0,
            "swing_reaction_roll_kp": 0.00,
            "swing_reaction_roll_rate_kd": 0.08,
            "swing_reaction_max_offset": 0.08,
            "swing_reaction_max_vel": 0.8,
            "swing_reaction_tau": 0.10,
            "lift_height": 0.14,
            "swing_clearance_min": 0.05,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
            "lift_duration": 0.6,
            "com_shift_y_offset": 0.0,
            "com_shift_duration": 3.0,
        },
        {
            "name": "stiffer_roll_less_sat",
            "W_RollAngularMomentum": 2500,
            "W_SwingLegRollMomentum": 1000,
            "W_CentroidalForce": 200,
            "W_JointAcc": 60,
            "W_TorsoYawJointAcc": 3000,
            "roll_kp": 250,
            "roll_kd": 35,
            "roll_angle_kp": 2.0,
            "roll_rate_kd": 2.0,
            "roll_max_rate": 800,
            "lateral_contact_kd": 50,
            "pitch_accel_kp": 6.0,
            "pitch_accel_kd": 1.0,
            "pitch_max_lin_acc": 4.0,
            "swing_leg_roll_enabled": True,
            "swing_leg_roll_scale": -0.25,
            "swing_leg_roll_max_rate": 300,
            "swing_reaction_enabled": True,
            "swing_reaction_sign": -1.0,
            "lift_height": 0.14,
            "swing_clearance_min": 0.05,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
            "lift_duration": 0.6,
            "com_shift_y_offset": -0.02,
            "com_shift_duration": 3.0,
        },
        {
            "name": "force_dominant",
            "W_RollAngularMomentum": 1000,
            "W_SwingLegRollMomentum": 500,
            "W_CentroidalForce": 400,
            "W_Centroidal": 25,
            "W_JointAcc": 20,
            "roll_kp": 150,
            "roll_kd": 20,
            "roll_angle_kp": 1.0,
            "roll_rate_kd": 0.5,
            "roll_max_rate": 500,
            "pitch_accel_kp": 4.0,
            "pitch_accel_kd": 0.5,
            "pitch_max_lin_acc": 2.0,
            "swing_leg_roll_enabled": False,
            "swing_leg_roll_scale": -0.25,
            "swing_leg_roll_max_rate": 100,
            "swing_reaction_enabled": False,
            "lift_height": 0.14,
            "swing_clearance_min": 0.05,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
            "lift_duration": 0.8,
            "com_shift_y_offset": 0.0,
            "com_shift_duration": 4.0,
        },
        {
            "name": "upper_body_quiet",
            "W_RollAngularMomentum": 1500,
            "W_SwingLegRollMomentum": 3000,
            "W_CentroidalForce": 200,
            "W_JointAcc": 120,
            "W_TorsoYawJointAcc": 7000,
            "orientation_kp_xy": 100,
            "orientation_kd_xy": 10,
            "roll_kp": 250,
            "roll_kd": 35,
            "roll_angle_kp": 2.0,
            "roll_rate_kd": 2.0,
            "pitch_accel_kp": 6.0,
            "pitch_accel_kd": 1.0,
            "pitch_max_lin_acc": 4.0,
            "swing_kp_y": 50,
            "swing_kd_y": 5,
            "swing_leg_roll_enabled": True,
            "swing_leg_roll_scale": -0.5,
            "swing_leg_roll_max_rate": 300,
            "swing_reaction_enabled": True,
            "swing_reaction_sign": -1.0,
            "lift_height": 0.14,
            "swing_clearance_min": 0.05,
            "swing_clearance_kp": 800,
            "swing_clearance_kd": 60,
            "swing_clearance_max_acc": 120,
            "lift_duration": 0.6,
            "com_shift_y_offset": -0.02,
            "com_shift_duration": 3.0,
        },
    ]
    for candidate in handpicked[: max(0, trials - 1)]:
        yield candidate

    remaining = max(0, trials - 1 - len(handpicked))
    keys = list(grids)
    for i in range(remaining):
        candidate = {"name": f"random_{i:03d}"}
        for key in keys:
            candidate[key] = rng.choice(grids[key])
        yield candidate


def main():
    parser = argparse.ArgumentParser(description="Run headless UnicycleCtrl weight sweeps.")
    parser.add_argument("--binary", default=str(ROOT / "build" / "mahru_ctrl"))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--duration", type=float, default=7.6)
    parser.add_argument("--log-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--stop-com-z", type=float, default=0.45)
    parser.add_argument("--stop-roll-deg", type=float, default=45.0)
    parser.add_argument("--stop-pitch-deg", type=float, default=45.0)
    parser.add_argument("--max-wall-time", type=float, default=None)
    parser.add_argument("--progress-interval", type=float, default=-1.0)
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "tuning_unicycle"))
    parser.add_argument("--keep-best", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    original_weighted = WEIGHTED_CONFIG.read_text()
    original_fsm = FSM_CONFIG.read_text()
    rows = []

    try:
        for trial_index, candidate in enumerate(candidate_stream(args.trials, args.seed)):
            trial_dir = out_dir / f"{trial_index:03d}_{candidate['name']}"
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            trial_dir.mkdir(parents=True)

            weighted_text, fsm_text = apply_candidate(
                original_weighted, original_fsm, candidate
            )
            WEIGHTED_CONFIG.write_text(weighted_text)
            FSM_CONFIG.write_text(fsm_text)
            (trial_dir / "config_weighted.yaml").write_text(weighted_text)
            (trial_dir / "fsm_UnicycleCtrl_config.yaml").write_text(fsm_text)

            cmd = [
                args.binary,
                "--headless",
                "--duration",
                str(args.duration),
                "--log-stride",
                str(args.log_stride),
                "--stop-com-z",
                str(args.stop_com_z),
                "--stop-roll-deg",
                str(args.stop_roll_deg),
                "--stop-pitch-deg",
                str(args.stop_pitch_deg),
                "--progress-interval",
                str(args.progress_interval),
                "--log-dir",
                str(trial_dir),
            ]
            if args.max_wall_time is not None:
                cmd.extend(["--max-wall-time", str(args.max_wall_time)])
            print(f"[{trial_index + 1}/{args.trials}] {candidate['name']}", flush=True)
            try:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                (trial_dir / "stdout.txt").write_text(result.stdout)
                (trial_dir / "stderr.txt").write_text(result.stderr)
                metrics = evaluate_log(trial_dir, args.duration)
                if result.returncode != 0:
                    metrics["ok"] = 0
                    metrics["failure"] = f"returncode_{result.returncode}"
                    metrics["score"] += 1e6
            except subprocess.TimeoutExpired as exc:
                (trial_dir / "stdout.txt").write_text(text_from_timeout_stream(exc.stdout))
                (trial_dir / "stderr.txt").write_text(text_from_timeout_stream(exc.stderr))
                metrics = {"ok": 0, "score": 1e9, "failure": "timeout"}

            row = {
                "trial": trial_index,
                "name": candidate["name"],
                "log_dir": str(trial_dir.relative_to(ROOT)),
                **{k: v for k, v in candidate.items() if k != "name"},
                **metrics,
            }
            rows.append(row)

            sorted_rows = sorted(rows, key=lambda r: float(r["score"]))
            print(
                "  score={:.4g} fall={} com_y={:.4g} roll={:.4g} hdot_act={:.4g} hdot_wbc={:.4g} sw_z={:.4g}".format(
                    float(row["score"]),
                    row.get("fall", ""),
                    float(row.get("rms_com_y_m", math.nan)),
                    float(row.get("rms_roll_deg", math.nan)),
                    float(row.get("rms_hdot_actual_err_norm", math.nan)),
                    float(row.get("rms_hdot_wbc_err_norm", math.nan)),
                    float(row.get("right_swing_min_z_m", math.nan)),
                ),
                flush=True,
            )
            print(f"  best so far: {sorted_rows[0]['trial']} {sorted_rows[0]['name']} score={float(sorted_rows[0]['score']):.4g}", flush=True)

            all_keys = sorted(set(itertools.chain.from_iterable(row.keys() for row in rows)))
            with results_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(rows)

        best = min(rows, key=lambda r: float(r["score"]))
        print("\nTop candidates:")
        for row in sorted(rows, key=lambda r: float(r["score"]))[: min(5, len(rows))]:
            print(
                "#{trial} {name}: score={score:.4g}, fall={fall}, com_y={com_y:.4g}, roll={roll:.4g}, hdot_act={hdot_act:.4g}, hdot_wbc={hdot_wbc:.4g}, sw_z={sw_z:.4g}".format(
                    trial=row.get("trial", ""),
                    name=row.get("name", ""),
                    score=float(row.get("score", math.nan)),
                    fall=row.get("fall", ""),
                    com_y=float(row.get("rms_com_y_m", math.nan)),
                    roll=float(row.get("rms_roll_deg", math.nan)),
                    hdot_act=float(row.get("rms_hdot_actual_err_norm", math.nan)),
                    hdot_wbc=float(row.get("rms_hdot_wbc_err_norm", math.nan)),
                    sw_z=float(row.get("right_swing_min_z_m", math.nan)),
                ),
                flush=True,
            )

        if args.keep_best:
            best_weighted = out_dir / f"{int(best['trial']):03d}_{best['name']}" / "config_weighted.yaml"
            best_fsm = out_dir / f"{int(best['trial']):03d}_{best['name']}" / "fsm_UnicycleCtrl_config.yaml"
            WEIGHTED_CONFIG.write_text(best_weighted.read_text())
            FSM_CONFIG.write_text(best_fsm.read_text())
            print(f"\nApplied best config from {best['log_dir']}")
    finally:
        if not args.keep_best:
            WEIGHTED_CONFIG.write_text(original_weighted)
            FSM_CONFIG.write_text(original_fsm)
            print("\nRestored original config files.")

    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
