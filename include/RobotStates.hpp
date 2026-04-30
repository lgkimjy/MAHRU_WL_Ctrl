#pragma once

#include <array>
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

    Eigen::Vector2d    p_ZMP = Eigen::Vector2d::Zero();
    Eigen::Vector2d    p_DCM = Eigen::Vector2d::Zero();

    std::array<Eigen::Vector3d, 4>                          p_C{};          // Contact Position w.r.t {I}
    std::array<Eigen::Vector3d, 4>                          pdot_C{};       // Contact Velocity w.r.t {I}
    std::array<Eigen::Matrix3d, 4>                          R_C{};          // Contact orientation w.r.t {I}
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jp_C{};         // Contact linear Jacobian
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jdotp_C{};      // Time derivative of contact linear Jacobian
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jr_C{};         // Contact angular Jacobian
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jdotr_C{};      // Time derivative of contact angular Jacobian
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jp_C_prev{};    // Contact Position
    std::array<Eigen::Matrix<double, 3, mahru::nDoF>, 4>    Jr_C_prev{};    // Contact Position

    // Joint States
    Eigen::VectorXd     jpos;
    Eigen::VectorXd     jvel;

    explicit RobotFeedback(int nv = mahru::nDoF)
        : qpos(Eigen::VectorXd::Zero(nv + 1)),
          qvel(Eigen::VectorXd::Zero(nv)),
          jpos(Eigen::VectorXd::Zero(mahru::num_act_joint)),
          jvel(Eigen::VectorXd::Zero(mahru::num_act_joint))
    {
        for (int i = 0; i < 4; ++i) {
            p_C[i].setZero();
            pdot_C[i].setZero();
            R_C[i].setIdentity();
            Jp_C[i].setZero();
            Jdotp_C[i].setZero();
            Jr_C[i].setZero();
            Jdotr_C[i].setZero();
            Jp_C_prev[i].setZero();
            Jr_C_prev[i].setZero();
        }
    }
};

// --- Commands & actuator outputs: motion targets + torque (not "motor struct" only) ---
struct RobotCtrl {
    Eigen::Vector3d lin_vel_d = Eigen::Vector3d::Zero(); // desired linear velocity w.r.t. base frame
    Eigen::Vector3d ang_vel_d = Eigen::Vector3d::Zero(); // desired angular velocity w.r.t. base frame
    int gait_type = 0; // 0: stand, 1: line walk, 2: point walk, 3: line walk DSP, 4: point walk DSP, 7: slide
    Eigen::Matrix<int, 4, 10> contact_schedule = Eigen::Matrix<int, 4, 10>::Ones();

    Eigen::VectorXd jpos_d;
    Eigen::VectorXd jvel_d;
    Eigen::VectorXd torq_d;
    Eigen::Vector3d p_CoM_d = Eigen::Vector3d::Zero();
    Eigen::Vector3d pdot_CoM_d = Eigen::Vector3d::Zero();
    double roll_momentum_rate_d = 0.0;
    double roll_momentum_rate_actual = 0.0;
    double roll_momentum_rate_wbc = 0.0;
    double roll_momentum_rate_wbc_error = 0.0;
    double swing_leg_roll_momentum_rate_d = 0.0;
    double swing_leg_roll_momentum_rate_wbc = 0.0;
    double roll_momentum_y_err = 0.0;
    double roll_momentum_ydot_err = 0.0;
    double roll_momentum_height = 0.0;
    double unicycle_state_time = 0.0;
    double right_foot_lift_phase = 0.0;
    double swing_leg_reaction_offset = 0.0;
    double swing_leg_reaction_vel = 0.0;
    double swing_lateral_acceleration_d = 0.0;
    double single_wheel_pitch = 0.0;
    double single_wheel_pitch_rate = 0.0;
    double single_wheel_lin_vel_d = 0.0;
    double single_wheel_lin_acc_d = 0.0;
    double single_wheel_com_offset_d = 0.0;
    double single_wheel_phase = 0.0;
    double single_wheel_stance_qdot_d = 0.0;
    double single_wheel_stance_qdot = 0.0;
    double single_wheel_lateral_vel = 0.0;

    explicit RobotCtrl(int = mahru::nDoF)
        : jpos_d(Eigen::VectorXd::Zero(mahru::num_act_joint)),
          jvel_d(Eigen::VectorXd::Zero(mahru::num_act_joint)),
          torq_d(Eigen::VectorXd::Zero(mahru::num_act_joint)) {}
};

// --- Parameters from config (PD, nominal posture); joint order = RobotDefinition::kJointNames / joint_names ---
struct RobotParam {
    Eigen::VectorXd Kp;
    Eigen::VectorXd Kd;

    explicit RobotParam(int = mahru::nDoF)
        : Kp(Eigen::VectorXd::Zero(mahru::num_act_joint)),
          Kd(Eigen::VectorXd::Zero(mahru::num_act_joint)) {}
};

struct RobotData {
    RobotFeedback fbk;
    RobotCtrl ctrl;
    RobotParam param;
    
    // n_q : number of generalized coordinates)
    // n_v : number of generalized velocities
    // n_j : number of joints
    explicit RobotData(int nv = mahru::nDoF) : fbk(nv), ctrl(nv), param(nv) {}
};
