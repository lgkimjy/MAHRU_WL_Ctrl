#include "Controller/ConvexMPC/LocoCtrl.hpp"

#include <algorithm>
#include <cmath>

LocoCtrl::LocoCtrl() :
    stand(12, Eigen::Vector4i(0, 0, 0, 0), Eigen::Vector4i(12, 12, 12, 12), "Stand"),
    linewalk(12, Eigen::Vector4i(0, 0, 6, 6), Eigen::Vector4i(6, 6, 6, 6), "LineWalk"),
    pointwalk(12, Eigen::Vector4i(0, 0, 0, 6), Eigen::Vector4i(0, 6, 0, 6), "PointWalk"),
    linewalk2(16, Eigen::Vector4i(3, 3, 11, 11), Eigen::Vector4i(10, 10, 10, 10), "LineWalkDSP"),
    pointwalk2(16, Eigen::Vector4i(0, 3, 0, 11), Eigen::Vector4i(0, 10, 0, 10), "PointWalkDSP"),
    run(12, Eigen::Vector4i(0, 0, 0, 0), Eigen::Vector4i(12, 12, 12, 12), "Run"),
    slide(20, Eigen::Vector4i(3, 3, 13, 13), Eigen::Vector4i(14, 14, 14, 14), "Sliding")
{
    optTraj.set_mid_air_height(0.10);
    optTraj.set_costs(1e1, 1e1, 1e0, 1e-6);
    previewTraj.set_mid_air_height(0.10);
    previewTraj.set_costs(1e1, 1e1, 1e0, 1e-6);
    reset(0);
}

void LocoCtrl::reset(int gait_type)
{
    iterationCounter = 0;
    gait_cur = &stand;
    prev_gait = &stand;
    StateMachine = DOUBLE_STANCE;
    prevStateMachine = DOUBLE_STANCE;
    cur_sw_time = 0.0;
    swing_contact_index = -1;

    left_foot_position.setZero();
    left_foot_velocity.setZero();
    left_foot_acceleration.setZero();
    right_foot_position.setZero();
    right_foot_velocity.setZero();
    right_foot_acceleration.setZero();
    p_footplacement_target.setZero();
    swing_preview.clear();
    lprevious_support_foot_position_.setZero();
    rprevious_support_foot_position_.setZero();
    resetMpc();

    ContactSchedule schedule = ContactSchedule::Ones();
    compute_contactSequence(gait_type, schedule);
}

bool LocoCtrl::compute_contactSequence(int gaitType, ContactSchedule& schedule)
{
    Gait *gait;
    if(gaitType == 0) {
        gait = &stand;
    } else if(gaitType == 1) {
        gait = &linewalk;
    } else if(gaitType == 2) {
        gait = &pointwalk;
    } else if(gaitType == 3) {
        gait = &linewalk2;
    } else if(gaitType == 4) {
        gait = &pointwalk2;
    } else if(gaitType == 5 || gaitType == 6) {
        gait = &run;
    } else if(gaitType == 7) {
        gait = &slide;
    } else {
        gait = &stand;
    }

    const bool gait_changed = gait_cur != gait;
    if(gait_changed) {
        iterationCounter = 0;
    }
    gait_cur = gait;

    int iterationsBetweenMPC = 50;
    gait->setIterations(iterationsBetweenMPC, iterationCounter);
    mpcTable = gait->mpc_gait();

    const int gait_segments = gait->get_nMPC_segments();
    for(int horizon = 0; horizon < ConvexMpc::kPlanHorizon; horizon++) {
        const int segment = std::min(horizon, gait_segments - 1);
        for(int foot = 0; foot < ConvexMpc::kNumContacts; foot++) {
            schedule(foot, horizon) = mpcTable[segment * ConvexMpc::kNumContacts + foot];
        }
    }

    if(schedule(1, 0) == 1 && schedule(3, 0) == 1) {
        StateMachine = DOUBLE_STANCE;
    } else if(schedule(1, 0) == 0 && schedule(3, 0) == 1) {
        StateMachine = LEFT_CONTACT;
    } else if(schedule(1, 0) == 1 && schedule(3, 0) == 0) {
        StateMachine = RIGHT_CONTACT;
    } else {
        StateMachine = DOUBLE_STANCE;
    }

    if(StateMachine != prevStateMachine) {
        cur_sw_time = 0.0;
        prevStateMachine = StateMachine;
    }
    return gait_changed;
}

void LocoCtrl::compute_nextfoot(
    const RobotData& robot,
    const Eigen::Vector3d& p_CoM,
    const Eigen::Vector3d& pdot_CoM,
    stateMachineTypeDef FSM,
    double cur_sw_time)
{
    if(FSM == DOUBLE_STANCE) {
        swing_contact_index = -1;
        swing_preview.clear();
        return;
    }

    Eigen::Matrix3d K_raibert;
    double K_cp = 1.0;

    K_raibert.setZero();
    K_raibert(0, 0) = 0.15;
    K_raibert(1, 1) = 0.15;

    Eigen::Vector3d offset;
    double pelvis_width = 0.257;

    if(gait_cur == &linewalk || gait_cur == &linewalk2 || gait_cur == &slide) {
        offset << (FSM == LEFT_CONTACT ? 0.1 : -0.1), pelvis_width/2, 0;
        prev_gait = gait_cur;
    } else if(gait_cur == &pointwalk || gait_cur == &pointwalk2) {
        offset << 0, pelvis_width/2, 0;
        prev_gait = gait_cur;
    } else if(gait_cur == &stand) {
        if(prev_gait == &linewalk || prev_gait == &linewalk2 || prev_gait == &slide) {
            offset << (FSM == LEFT_CONTACT ? 0.1 : -0.1), pelvis_width/2, 0;
        } else if(prev_gait == &pointwalk || prev_gait == &pointwalk2) {
            offset << 0, pelvis_width/2, 0;
        } else {
            offset << 0, pelvis_width/2, 0;
        }
    } else {
        offset << 0, pelvis_width/2, 0;
    }

    offset = robot.fbk.R_B * offset;
    if(FSM == LEFT_CONTACT)
    {
        p_footplacement_target =
            -offset + p_CoM
            + 0.3 / 2 * pdot_CoM
            + K_raibert * (pdot_CoM - robot.fbk.R_B * robot.ctrl.lin_vel_d)
            + (K_cp * sqrt(0.7/9.8)) * robot.ctrl.ang_vel_d.cross(pdot_CoM);
        p_footplacement_target(2) = 0.0;
    }
    else if(FSM == RIGHT_CONTACT)
    {
        p_footplacement_target =
            offset + p_CoM
            + 0.3 / 2 * pdot_CoM
            + K_raibert * (pdot_CoM - robot.fbk.R_B * robot.ctrl.lin_vel_d)
            + (K_cp * sqrt(0.7/9.8)) * robot.ctrl.ang_vel_d.cross(pdot_CoM);
        p_footplacement_target(2) = 0.0;
    }

    if(cur_sw_time < 0.002)
    {
        if(FSM == LEFT_CONTACT)
        {
            rprevious_support_foot_position_ = robot.fbk.p_C[1];
            rprevious_support_foot_position_(2) = 0.0;
            right_foot_position = rprevious_support_foot_position_;
            right_foot_velocity.setZero();
            right_foot_acceleration.setZero();
        }
        else if(FSM == RIGHT_CONTACT)
        {
            lprevious_support_foot_position_ = robot.fbk.p_C[3];
            lprevious_support_foot_position_(2) = 0.0;
            left_foot_position = lprevious_support_foot_position_;
            left_foot_velocity.setZero();
            left_foot_acceleration.setZero();
        }
    }

    double init_step_time = 0.0;
    double curr_step_time = cur_sw_time;
    double step_duration = 0.3;
    if(FSM == LEFT_CONTACT)
    {
        swing_contact_index = 1;
        if(curr_step_time < step_duration - 0.01)
        {
            optTraj.compute(rprevious_support_foot_position_,
                            right_foot_position, right_foot_velocity, right_foot_acceleration,
                            p_footplacement_target,
                            init_step_time, curr_step_time, step_duration);
        }
        optTraj.get_next_state(curr_step_time + 0.001,
                               right_foot_position, right_foot_velocity, right_foot_acceleration);

        left_foot_position = robot.fbk.p_C[3];
        left_foot_position(2) = 0.0;
        left_foot_velocity.setZero();
        left_foot_acceleration.setZero();
    }
    else if(FSM == RIGHT_CONTACT)
    {
        swing_contact_index = 3;
        if(curr_step_time < step_duration - 0.01)
        {
            optTraj.compute(lprevious_support_foot_position_,
                            left_foot_position, left_foot_velocity, left_foot_acceleration,
                            p_footplacement_target,
                            init_step_time, curr_step_time, step_duration);
        }
        optTraj.get_next_state(curr_step_time + 0.001,
                               left_foot_position, left_foot_velocity, left_foot_acceleration);

        right_foot_position = robot.fbk.p_C[1];
        right_foot_position(2) = 0.0;
        right_foot_velocity.setZero();
        right_foot_acceleration.setZero();
    }

    swing_preview.clear();
    if(curr_step_time < step_duration - 0.01) {
        Eigen::Vector3d current_pos = swingPosition();
        Eigen::Vector3d current_vel = swingVelocity();
        Eigen::Vector3d current_acc =
            FSM == LEFT_CONTACT ? right_foot_acceleration : left_foot_acceleration;
        const Eigen::Vector3d& start_pos =
            FSM == LEFT_CONTACT ? rprevious_support_foot_position_ : lprevious_support_foot_position_;
        if(previewTraj.compute(start_pos,
                               current_pos, current_vel, current_acc,
                               p_footplacement_target,
                               init_step_time, curr_step_time, step_duration)) {
            swing_preview.reserve(kSwingPreviewSize);
            for(int i = 0; i < kSwingPreviewSize; i++) {
                const double alpha = static_cast<double>(i) / (kSwingPreviewSize - 1);
                const double sample_time = curr_step_time + alpha * (step_duration - curr_step_time);
                Eigen::Vector3d point, velocity, acceleration;
                previewTraj.get_next_state(sample_time, point, velocity, acceleration);
                swing_preview.push_back(point);
            }
        }
    }
}

bool LocoCtrl::compute_grf(const ConvexMpc::Input& input, double dt)
{
    const bool solved = mpc_solver.update(input, dt);
    if (solved) {
        grf_mpc = mpc_solver.groundReactionForce();
    }
    return solved;
}

void LocoCtrl::resetMpc()
{
    mpc_solver.reset();
    grf_mpc.setZero();
}

void LocoCtrl::step()
{
    ++iterationCounter;
    if(StateMachine == DOUBLE_STANCE) {
        cur_sw_time = 0.0;
    } else if(cur_sw_time < kSwingDuration) {
        cur_sw_time += kControlDt;
    }
}

const Eigen::Vector3d& LocoCtrl::swingPosition() const
{
    return swing_contact_index == 1 ? right_foot_position : left_foot_position;
}

const Eigen::Vector3d& LocoCtrl::swingVelocity() const
{
    return swing_contact_index == 1 ? right_foot_velocity : left_foot_velocity;
}
