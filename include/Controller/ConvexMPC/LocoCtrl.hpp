#pragma once

#include <vector>

#include <Eigen/Dense>

#include "Controller/ConvexMPC/ConvexMpc.h"
#include "Controller/ConvexMPC/GaitGenerator.hpp"
#include "RobotStates.hpp"
#include "Trajectory/OnlineOptTraj.hpp"

class LocoCtrl
{
public:
    enum stateMachineTypeDef {
        DOUBLE_STANCE = 0,
        LEFT_CONTACT = 1,
        RIGHT_CONTACT = 2,
    };
    using ContactSchedule =
        Eigen::Matrix<int, ConvexMpc::kNumContacts, ConvexMpc::kPlanHorizon>;

    LocoCtrl();
    ~LocoCtrl() = default;

    void reset(int gait_type);
    bool compute_contactSequence(int gaitType, ContactSchedule& schedule);
    void compute_nextfoot(
        const RobotData& robot,
        const Eigen::Vector3d& p_CoM,
        const Eigen::Vector3d& pdot_CoM,
        stateMachineTypeDef FSM,
        double cur_sw_time);
    bool compute_grf(const ConvexMpc::Input& input, double dt);
    void resetMpc();
    void step();

    bool updateGaitSchedule(int gait_type, ContactSchedule& schedule)
    {
        return compute_contactSequence(gait_type, schedule);
    }
    void updateSwingFoot(
        const RobotData& robot,
        const Eigen::Vector3d& p_CoM,
        const Eigen::Vector3d& pdot_CoM)
    {
        compute_nextfoot(robot, p_CoM, pdot_CoM, StateMachine, cur_sw_time);
    }

    int swingContactIndex() const { return swing_contact_index; }
    const Eigen::Vector3d& swingPosition() const;
    const Eigen::Vector3d& swingVelocity() const;
    const Eigen::Vector3d& swingTarget() const { return p_footplacement_target; }
    const std::vector<Eigen::Vector3d>& swingPreview() const { return swing_preview; }
    const Eigen::Matrix<double, ConvexMpc::kForceDim, 1>& groundReactionForce() const
    {
        return grf_mpc;
    }
    const Eigen::Matrix<double, ConvexMpc::kStateDim * ConvexMpc::kPlanHorizon, 1>& predictedStates() const
    {
        return mpc_solver.predictedStates();
    }

    int iterationCounter = 0;
    int* mpcTable = nullptr;

    Gait *gait_cur = nullptr;
    Gait *prev_gait = nullptr;
    Gait stand, linewalk, pointwalk, linewalk2, pointwalk2, run, slide;
    ConvexMpc mpc_solver;
    OnlineOptTraj optTraj;
    OnlineOptTraj previewTraj;
    Eigen::Matrix<double, ConvexMpc::kForceDim, 1> grf_mpc =
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();

    Eigen::Vector3d left_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_footplacement_target = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> swing_preview;
    ContactSchedule contact_window = ContactSchedule::Ones();

    stateMachineTypeDef StateMachine = DOUBLE_STANCE;
    stateMachineTypeDef prevStateMachine = DOUBLE_STANCE;
    double cur_sw_time = 0.0;
    int swing_contact_index = -1;
    Eigen::Vector3d lprevious_support_foot_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d rprevious_support_foot_position_ = Eigen::Vector3d::Zero();

private:
    static constexpr double kControlDt = 0.001;
    static constexpr double kSwingDuration = 0.3;
    static constexpr int kSwingPreviewSize = 10;
};
