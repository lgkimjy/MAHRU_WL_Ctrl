#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

#include "RobotDefinition.hpp"

// --- Feedback (sensors / sim truth): log under e.g. "/fbk/..." for HDF5 ---
struct RobotFeedback {
    Eigen::VectorXd     qpos; // generalized coordinates
    Eigen::VectorXd     qvel; // generalized velocities
    
    // Position of base
    Eigen::Vector3d     p_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d     pdot_B = Eigen::Vector3d::Zero(); // linear velocity w.r.t world frame

    // Orientation of base
    Eigen::Quaterniond  quat_B = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d     R_B = Eigen::Matrix3d::Identity();
    Eigen::Vector3d     varphi_B = Eigen::Vector3d::Zero();
    Eigen::Vector3d     omega_B = Eigen::Vector3d::Zero();

    // Position of CoM
    Eigen::Vector3d     p_CoM = Eigen::Vector3d::Zero();
    Eigen::Vector3d     pdot_CoM = Eigen::Vector3d::Zero();

    // Joint States
    Eigen::VectorXd     jpos;
    Eigen::VectorXd     jvel;

    explicit RobotFeedback(int nv = robot_name::nDoF)
        : qpos(Eigen::VectorXd::Zero(nv + 1)),
          qvel(Eigen::VectorXd::Zero(nv)),
          jpos(Eigen::VectorXd::Zero(robot_name::num_act_joint)),
          jvel(Eigen::VectorXd::Zero(robot_name::num_act_joint)) {}
};

// --- Commands & actuator outputs: motion targets + torque (not "motor struct" only) ---
struct RobotCtrl {
    Eigen::Vector3d lin_vel_d = Eigen::Vector3d::Zero(); // desired linear velocity w.r.t. base frame
    Eigen::Vector3d ang_vel_d = Eigen::Vector3d::Zero(); // desired angular velocity w.r.t. base frame

    Eigen::VectorXd jpos_d;
    Eigen::VectorXd jvel_d;
    Eigen::VectorXd torq_d;

    explicit RobotCtrl(int = robot_name::nDoF)
        : jpos_d(Eigen::VectorXd::Zero(robot_name::num_act_joint)),
          jvel_d(Eigen::VectorXd::Zero(robot_name::num_act_joint)),
          torq_d(Eigen::VectorXd::Zero(robot_name::num_act_joint)) {}
};

// --- Parameters from config (PD, nominal posture); joint order = RobotDefinition::kJointNames / joint_names ---
struct RobotParam {
    Eigen::VectorXd Kp;
    Eigen::VectorXd Kd;

    explicit RobotParam(int = robot_name::nDoF)
        : Kp(Eigen::VectorXd::Zero(robot_name::num_act_joint)),
          Kd(Eigen::VectorXd::Zero(robot_name::num_act_joint)) {}
};

struct RobotData {
    RobotFeedback fbk;
    RobotCtrl ctrl;
    RobotParam param;
    
    // n_q : number of generalized coordinates)
    // n_v : number of generalized velocities
    // n_j : number of joints
    explicit RobotData(int nv = robot_name::nDoF) : fbk(nv), ctrl(nv), param(nv) {}
};
