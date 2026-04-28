#include "MJSimulationBridge.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

SimulationBridge::SimulationBridge(const std::string& scene_file)
    : SimulationInterface(scene_file),
      robot_(nDoF),
      state_machine_(robot_)
{
    const auto now = std::chrono::system_clock::now();
    const auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");

    log_dir = CMAKE_SOURCE_DIR "/logs/" + ss.str() + "/";
    std::filesystem::create_directories(log_dir);
    logger = std::make_unique<HDF5Logger>(log_dir + log_file_name);

    std::cout << "[ SimulationBridge ] Constructed" << std::endl;
}

void SimulationBridge::Initialize()
{
    state_machine_.setVisualizer(&mjSim_->traj_viz_util_);
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
}

void SimulationBridge::UpdateUserInput()
{
    robot_.ctrl.lin_vel_d = mjSim_->lin_vel_d;
    robot_.ctrl.ang_vel_d = mjSim_->ang_vel_d;    
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
