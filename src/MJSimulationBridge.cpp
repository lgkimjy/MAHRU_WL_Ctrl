#include "MJSimulationBridge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
constexpr double kRadToDeg = 57.29577951308232;
constexpr int kRobotRootBody = 1;

std::pair<double, double> baseRollPitchDeg(const mjData* data)
{
    const double w = data->qpos[3];
    const double x = data->qpos[4];
    const double y = data->qpos[5];
    const double z = data->qpos[6];

    const double sinr_cosp = 2.0 * (w * x + y * z);
    const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    const double roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (w * y - z * x);
    const double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));
    return {roll * kRadToDeg, pitch * kRadToDeg};
}

double robotComZ(const mjModel* model, const mjData* data)
{
    if (model->nbody > kRobotRootBody) {
        return data->subtree_com[3 * kRobotRootBody + 2];
    }
    return data->qpos[2];
}
}

SimulationBridge::SimulationBridge(const std::string& scene_file,
                                   const std::string& log_dir_override,
                                   bool headless)
    : SimulationInterface(scene_file, !headless),
      robot_(nDoF),
      state_machine_(robot_)
{
    if (log_dir_override.empty()) {
        const auto now = std::chrono::system_clock::now();
        const auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        log_dir = CMAKE_SOURCE_DIR "/logs/" + ss.str() + "/";
    } else {
        log_dir = log_dir_override;
        if (!log_dir.empty() && log_dir.back() != '/') {
            log_dir += "/";
        }
    }
    std::filesystem::create_directories(log_dir);
    logger = std::make_unique<HDF5Logger>(log_dir + log_file_name);

    std::cout << "[ SimulationBridge ] Constructed" << std::endl;
    std::cout << "[ SimulationBridge ] log_dir: " << log_dir << std::endl;
}

void SimulationBridge::RunHeadless(double duration_s,
                                   int log_stride,
                                   double stop_com_z,
                                   double stop_roll_deg,
                                   double stop_pitch_deg,
                                   double max_wall_time_s,
                                   double progress_interval_s)
{
    log_stride = std::max(1, log_stride);

    char load_error[kErrorLength] = "";
    mjModel_ = mj_loadXML(filename_, nullptr, load_error, kErrorLength);
    if (!mjModel_) {
        throw std::runtime_error(std::string("failed to load MuJoCo model: ") + load_error);
    }
    if (load_error[0]) {
        std::cout << "[ SimulationBridge ][ RunHeadless ] model warning: "
                  << load_error << std::endl;
    }

    mjData_ = mj_makeData(mjModel_);
    if (!mjData_) {
        throw std::runtime_error("failed to allocate MuJoCo data");
    }

    mj_forward(mjModel_, mjData_);

    std::cout << "--------------------------------" << std::endl;
    std::cout << "[ SimulationBridge ][ RunHeadless ] Start Simulation" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    Initialize();

    const auto wall_start = std::chrono::steady_clock::now();
    double next_progress_time = progress_interval_s > 0.0 ? 0.0 : duration_s + 1.0;
    int step_count = 0;
    while (mjData_->time < duration_s) {
        UpdateSystemObserver();
        UpdateUserInput();
        UpdateControlCommand();
        if (step_count % log_stride == 0) {
            UpdateSystemVisualInfo();
        }

        mj_step(mjModel_, mjData_);
        ++step_count;
        const auto [roll_deg, pitch_deg] = baseRollPitchDeg(mjData_);
        const double com_z = robotComZ(mjModel_, mjData_);
        const double wall_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        if (progress_interval_s > 0.0 && mjData_->time >= next_progress_time) {
            std::cout << "[ SimulationBridge ][ RunHeadless ] progress t="
                      << mjData_->time << " / " << duration_s
                      << " s, wall=" << wall_elapsed
                      << " s, com_z=" << com_z
                      << ", roll=" << roll_deg
                      << " deg, pitch=" << pitch_deg << " deg" << std::endl;
            next_progress_time += progress_interval_s;
        }
        if (max_wall_time_s > 0.0 && wall_elapsed > max_wall_time_s) {
            std::cout << "[ SimulationBridge ][ RunHeadless ] Wall-clock stop at t="
                      << mjData_->time << " s, wall=" << wall_elapsed
                      << " s, com_z=" << com_z
                      << ", roll=" << roll_deg
                      << " deg, pitch=" << pitch_deg << " deg" << std::endl;
            break;
        }
        if ((stop_com_z > 0.0 && com_z < stop_com_z) ||
            (stop_roll_deg > 0.0 && std::abs(roll_deg) > stop_roll_deg) ||
            (stop_pitch_deg > 0.0 && std::abs(pitch_deg) > stop_pitch_deg)) {
            std::cout << "[ SimulationBridge ][ RunHeadless ] Early stop at t="
                      << mjData_->time << " s, com_z=" << com_z
                      << ", roll=" << roll_deg
                      << " deg, pitch=" << pitch_deg << " deg" << std::endl;
            break;
        }
        if (const char* message = Diverged(mjModel_->opt.disableflags, mjData_)) {
            std::cout << "[ SimulationBridge ][ RunHeadless ] Diverged: "
                      << message << std::endl;
            break;
        }
    }

    UpdateSystemObserver();
    UpdateSystemVisualInfo();

    logger.reset();
    std::cout << "[ SimulationBridge ][ RunHeadless ] finished at t="
              << mjData_->time << " s" << std::endl;
}

void SimulationBridge::Initialize()
{
    if (mjSim_) {
        state_machine_.setVisualizer(&mjSim_->traj_viz_util_);
    }
    state_machine_.initialize();
    std::cout << "[ SimulationBridge ] Initialized" << std::endl;
}

void SimulationBridge::UpdateSystemObserver()
{
    if (!mjData_) return;
    
	/////	Position vector of floating-base body w.r.t {I}
    robot_.fbk.p_B(0) = mjData_->qpos[0];
    robot_.fbk.p_B(1) = mjData_->qpos[1];
    robot_.fbk.p_B(2) = mjData_->qpos[2];

    /////   Orientation of floating-base body expressed in {I} (quaternion)
    robot_.fbk.quat_B.w() = mjData_->qpos[3];
    robot_.fbk.quat_B.x() = mjData_->qpos[4];
    robot_.fbk.quat_B.y() = mjData_->qpos[5];
    robot_.fbk.quat_B.z() = mjData_->qpos[6];
    robot_.fbk.R_B = robot_.fbk.quat_B.normalized().toRotationMatrix();

    //////	Linear velocity of floating-base body expressed in {I}
    robot_.fbk.pdot_B(0) = mjData_->qvel[0];
    robot_.fbk.pdot_B(1) = mjData_->qvel[1];
    robot_.fbk.pdot_B(2) = mjData_->qvel[2];

    //////	Angular velocity of floating-base body expressed in {B}
    robot_.fbk.varphi_B(0) = mjData_->qvel[3];
    robot_.fbk.varphi_B(1) = mjData_->qvel[4];
    robot_.fbk.varphi_B(2) = mjData_->qvel[5];
    //////	Angular velocity of floating-base body expressed in {I}
    robot_.fbk.omega_B = robot_.fbk.R_B * robot_.fbk.varphi_B;

    //////  Joint positions and velocities
    for (int i = 0; i < num_act_joint; ++i) {
        robot_.fbk.jpos(i) = mjData_->qpos[jnt_mapping_idx[i] + 7];
        robot_.fbk.jvel(i) = mjData_->qvel[jnt_mapping_idx[i] + 6];
    }

    //////  Generalized coordinates
    robot_.fbk.qpos.segment<3>(0) = robot_.fbk.p_B;
    robot_.fbk.qpos(3) = robot_.fbk.quat_B.w();
    robot_.fbk.qpos(4) = robot_.fbk.quat_B.x();
    robot_.fbk.qpos(5) = robot_.fbk.quat_B.y();
    robot_.fbk.qpos(6) = robot_.fbk.quat_B.z();
    robot_.fbk.qpos.tail(num_act_joint) = robot_.fbk.jpos;

    //////  Generalized velocities
    robot_.fbk.qvel.segment<3>(0) = robot_.fbk.pdot_B;
    robot_.fbk.qvel.segment<3>(3) = robot_.fbk.omega_B; // or varphi_B
    robot_.fbk.qvel.tail(num_act_joint) = robot_.fbk.jvel;

    if (mjModel_->nbody > kRobotRootBody) {
        for (int i = 0; i < 3; ++i) {
            robot_.fbk.p_CoM(i) = mjData_->subtree_com[3 * kRobotRootBody + i];
            robot_.fbk.pdot_CoM(i) = mjData_->subtree_linvel[3 * kRobotRootBody + i];
        }
    }
}

void SimulationBridge::UpdateUserInput()
{
    if (mjSim_) {
        robot_.ctrl.lin_vel_d = mjSim_->lin_vel_d;
        robot_.ctrl.ang_vel_d = mjSim_->ang_vel_d;
        robot_.ctrl.gait_type = mjSim_->gait_type;
    } else {
        robot_.ctrl.lin_vel_d.setZero();
        robot_.ctrl.ang_vel_d.setZero();
        robot_.ctrl.gait_type = 0;
    }
}

void SimulationBridge::UpdateControlCommand()
{
    state_machine_.runState();

    for (int i = 0; i < num_act_joint; ++i) {
        mjData_->ctrl[jnt_mapping_idx[i]] = robot_.ctrl.torq_d(i);
    }
    // passive toe wheel joint
    mjData_->ctrl[13] = 0.0;
    mjData_->ctrl[19] = 0.0;
}

void SimulationBridge::UpdateSystemVisualInfo()
{
    LogStates();
}

void SimulationBridge::LogStates()
{
    if (logger && mjData_) {
        logger->log(mjData_->time, robot_);
    }
}
