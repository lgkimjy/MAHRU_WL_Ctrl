#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <OsqpEigen/OsqpEigen.h>

#include "types.h"
#include "Robot_States.h"
#include "ConvexMPC/ConvexMpc.h"
#include "ConvexMPC/GaitGenerator.hpp"
#include "Trajectory/JointTrajectory.h"
#include "Trajectory/OnlineOptTraj.hpp"

class LocoCtrl
{
private:
public:
    LocoCtrl();
    ~LocoCtrl() {};

    Eigen::Vector<double, MPC_STATE_DIM>                     mpc_states;
    Eigen::Vector<double, MPC_STATE_DIM * PLAN_HORIZON>      mpc_states_d;

    int iterationCounter;
    int *mpcTable;

    Gait                    *gait_cur;
    Gait                    *prev_gait;
    Gait                    stand, linewalk, pointwalk, linewalk2, pointwalk2, run, slide;
    ConvexMpc               mpc_solver;
    OsqpEigen::Solver       solver;     // mpc-solver
    OnlineOptTraj           optTraj;
    
    CP2P_Traj<DOF3, double> sliding_traj;
    Eigen::Vector3d         p_whl_slide_ref;
    Eigen::Vector3d         p_whl_slide;
    Eigen::Vector3d         p_whl_slide_d;
    Eigen::Vector3d         pdot_whl_slide_d;
    Eigen::Vector3d         pddot_whl_slide_d;

    Eigen::Vector3d left_foot_position, left_foot_velocity, left_foot_acceleration;
    Eigen::Vector3d right_foot_position, right_foot_velocity, right_foot_acceleration;

    void compute_contactSequence(RobotState &state, gaitTypeDef gaitType, double dt, stateMachineTypeDef &FSM);
    void calcReference(RobotState &state, double mpc_dt);
    auto compute_grf(RobotState &state) -> Eigen::Vector<double, MPC_STATE_DIM * PLAN_HORIZON>;
    void compute_nextfoot(RobotState &state, stateMachineTypeDef &FSM);
    void compute_nextfoot(RobotState &state, stateMachineTypeDef &FSM, double cur_sw_time);

    const Eigen::Vector3d &get_desired_left_foot_position()      { return left_foot_position; }
    const Eigen::Vector3d &get_desired_left_foot_velocity()     { return left_foot_velocity; }
    const Eigen::Vector3d &get_desired_left_foot_acceleration() { return left_foot_acceleration; }
    const Eigen::Vector3d &get_desired_right_foot_position()     { return right_foot_position; }
    const Eigen::Vector3d &get_desired_right_foot_velocity()     { return right_foot_velocity; }
    const Eigen::Vector3d &get_desired_right_foot_acceleration() { return right_foot_acceleration; }
    // const Eigen::Vector3d &get_next_support_foot_position()      { return next_support_foot_position_; }
    // const Eigen::Vector3d &get_current_support_foot_position()   { return current_support_foot_position_; }
};