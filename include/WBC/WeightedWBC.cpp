#include "WBC/WeightedWBC.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <OsqpEigen/OsqpEigen.h>

WeightedWBC::WeightedWBC()
{
    std::cout << "WeightedWBC Constructor" << std::endl;

    loadWeightGain();
    Sf_ = Eigen::MatrixXd::Zero(mahru::nDoF, mahru::nDoF);
    Sf_.block<DOF6, DOF6>(0, 0) = Eigen::Matrix<double, DOF6, DOF6>::Identity();
}

WeightedWBC::Output WeightedWBC::update(
    CARBML& robot,
    const RobotData& state,
    const Input& input)
{
    robot_ = &robot;
    state_ = &state;
    input_ = input;

    reconfigStates();
    configureContacts();
    diagnostics_ = Diagnostics{};
    diagnostics_.num_contact_points = numContactPoints_;
    diagnostics_.contact_jacobian_norm = Jp_contact_.norm();
    diagnostics_.contact_jacobian_max = Jp_contact_.cwiseAbs().maxCoeff();
    diagnostics_.swing_jacobian_norm = Jp_sw_.norm();

    const WBCTask constraints = formulateConstraints();
    const WBCTask weightedTask = formulateWeightedTask();

    Eigen::VectorXd qpSol;
    Output output;
    output.solved = solveQP(constraints, weightedTask, qpSol);
    if (!output.solved || qpSol.size() != kNumDecisionVars) {
        return output;
    }

    output.xiddot_d = qpSol.head<mahru::nDoF>();
    output.grfs_d = qpSol.segment<ConvexMpc::kForceDim>(mahru::nDoF);

    const Eigen::Matrix<double, mahru::nDoF, 1> generalized_torque =
        robot_->M_mat * output.xiddot_d
        + robot_->C_mat * robot_->xidot
        + robot_->g_vec
        - Jp_contact_.transpose() * (output.grfs_d + input_.grfs_mpc);

    output.torq_ff = generalized_torque.tail<mahru::num_act_joint>();
    diagnostics_.solved = true;
    diagnostics_.xiddot_max = output.xiddot_d.cwiseAbs().maxCoeff();
    diagnostics_.base_xiddot_max = output.xiddot_d.head<DOF6>().cwiseAbs().maxCoeff();
    diagnostics_.joint_qddot_max =
        output.xiddot_d.tail<mahru::num_act_joint>().cwiseAbs().maxCoeff();
    Eigen::Index joint_qddot_index = 0;
    output.xiddot_d.tail<mahru::num_act_joint>().cwiseAbs().maxCoeff(&joint_qddot_index);
    diagnostics_.joint_qddot_max_index = static_cast<int>(joint_qddot_index);
    diagnostics_.delta_grf_max = output.grfs_d.cwiseAbs().maxCoeff();
    diagnostics_.torque_norm = output.torq_ff.norm();
    diagnostics_.torque_max = output.torq_ff.cwiseAbs().maxCoeff();
    diagnostics_.joint_qddot_limit = joint_qddot_limit_;
    return output;
}

void WeightedWBC::reconfigStates()
{
    numContactPoints_ = 0;
    for (int i = 0; i < ConvexMpc::kNumContacts; ++i) {
        if (state_->ctrl.contact_schedule(i, 0) != 0) {
            ++numContactPoints_;
        }
    }
}

void WeightedWBC::configureContacts()
{
    Jp_contact_.setZero();
    Jdotp_contact_.setZero();
    Jp_sw_.setZero();
    Jdotp_sw_.setZero();
    p_sw_.setZero();
    pdot_sw_.setZero();

    for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }
        Jp_contact_.block<DOF3, mahru::nDoF>(DOF3 * contact, 0) =
            state_->fbk.Jp_C[contact];
        Jdotp_contact_.block<DOF3, mahru::nDoF>(DOF3 * contact, 0) =
            state_->fbk.Jdotp_C[contact];
    }

    const int swing_contact = input_.swing_contact_index;
    if (swing_contact >= 0 && swing_contact < ConvexMpc::kNumContacts) {
        p_sw_ = state_->fbk.p_C[swing_contact];
        pdot_sw_ = state_->fbk.pdot_C[swing_contact];
        Jp_sw_ = state_->fbk.Jp_C[swing_contact];
        Jdotp_sw_ = state_->fbk.Jdotp_C[swing_contact];
    }
}

WBCTask WeightedWBC::formulateConstraints()
{
    return formulateFloatingBaseConstraint()
        + formulateContactNormalConstraint()
        + formulateSwingClearanceConstraint()
        + formulateSwingLateralClearanceConstraint()
        + formulateFrictionConeConstraint()
        + formulateAccelerationLimitConstraint();
}

WBCTask WeightedWBC::formulateWeightedTask()
{
    WBCTask task(kNumDecisionVars);

    task = task
        + formulateJointAccelerationTask(selectedJointsIdx_, kp_jacc_, kd_jacc_) * W_JointAcc_
        + formulateSlidingJointTask() * W_wheelAccel_
        + formulateLinearMotionTask() * W_centroidal_;

    if (input_.enable_centroidal_force_task) {
        task = task + formulateCentroidalForceTask() * W_centroidal_force_;
    }

    if (input_.enable_roll_angular_momentum_task) {
        task = task + formulateRollAngularMomentumTask() * W_roll_angular_momentum_;
    }

    if (input_.enable_swing_leg_roll_momentum_task && W_swing_leg_roll_momentum_ > 0.0) {
        task = task + formulateSwingLegRollMomentumTask() * W_swing_leg_roll_momentum_;
    }

    if (input_.enable_arm_roll_momentum_task && W_arm_roll_momentum_ > 0.0) {
        task = task + formulateArmRollMomentumTask() * W_arm_roll_momentum_;
    }

    if (input_.enable_swing_lateral_acceleration_task && W_swing_lateral_acceleration_ > 0.0) {
        task = task + formulateSwingLateralAccelerationTask() * W_swing_lateral_acceleration_;
    }

    if (W_torso_yaw_joint_acc_ > 0.0) {
        task = task + formulateTorsoYawJointAccelerationTask() * W_torso_yaw_joint_acc_;
    }

    if (input_.swing_contact_index >= 0 && numContactPoints_ < ConvexMpc::kNumContacts) {
        task = task + formulateSwingLegTask(kp_swing_, kd_swing_) * W_swingLeg_;
        if (input_.secondary_swing_contact_index >= 0) {
            task = task
                + formulateSwingContactTask(input_.secondary_swing_contact_index,
                                            input_.p_sw2_d,
                                            input_.pdot_sw2_d,
                                            kp_swing_,
                                            kd_swing_) * W_swingLeg_;
        }
        task = task + formulateSwingWheelTask() * W_wheelAccel_;
    }

    return task + formulateRegularizationTask(W_qddot_regul_, W_rf_regul_);
}

WBCTask WeightedWBC::formulateLinearMotionTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF4, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF4);

    const Eigen::Matrix3d errorMatrix = input_.R_B_d * robot_->R_B.transpose();
    const Eigen::AngleAxisd errorAngleAxis(errorMatrix);
    Eigen::Vector3d oriErr = errorAngleAxis.angle() * errorAngleAxis.axis();
    if (!oriErr.allFinite()) {
        oriErr.setZero();
    }

    a(0, 2) = 1.0;
    a(1, 3) = 1.0;
    a(2, 4) = 1.0;
    a(3, 5) = 1.0;
    b(0) = kp_CoM_(2, 2) * (input_.p_CoM_d(2) - robot_->p_CoM(2))
         + kd_CoM_(2, 2) * (input_.pdot_CoM_d(2) - robot_->pdot_CoM(2));
    b.segment<3>(1) =
        kp_R_ * oriErr
        + kd_omega_ * (input_.omega_B_d - robot_->omega_B)
        + input_.omegadot_B_ff;

    for (int i = 0; i < DOF4; ++i) {
        const double mask = std::clamp(input_.linear_motion_task_mask(i), 0.0, 1.0);
        a.row(i) *= mask;
        b(i) *= mask;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateCentroidalForceTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF6, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF6);

    const Eigen::Matrix3d errorMatrix = input_.R_B_d * robot_->R_B.transpose();
    const Eigen::AngleAxisd errorAngleAxis(errorMatrix);
    Eigen::Vector3d oriErr = errorAngleAxis.angle() * errorAngleAxis.axis();
    if (!oriErr.allFinite()) {
        oriErr.setZero();
    }

    const Eigen::Vector3d pddot_CoM_d =
        input_.pddot_CoM_ff
        + kp_CoM_ * (input_.p_CoM_d - robot_->p_CoM)
        + kd_CoM_ * (input_.pdot_CoM_d - robot_->pdot_CoM);
    const Eigen::Vector3d gravity(0.0, 0.0, robot_->getGravityConst());
    Eigen::Vector3d desired_force =
        robot_->getTotalMass() * (pddot_CoM_d + gravity);

    Eigen::Matrix3d I_G = Eigen::Matrix3d::Zero();
    for (int i = 0; i < static_cast<int>(mahru::NO_OF_BODY); ++i) {
        I_G += robot_->I_G_BCS[i]
             - robot_->body[i].get_mass()
               * Skew(robot_->rpos_lnk[i]) * Skew(robot_->rpos_lnk[i]);
    }
    const Eigen::Vector3d omegadot_B_d =
        input_.omegadot_B_ff
        + kp_R_ * oriErr
        + kd_omega_ * (input_.omega_B_d - robot_->omega_B);
    Eigen::Vector3d desired_moment = I_G * omegadot_B_d;
    desired_moment += input_.centroidal_moment_ff;

    Eigen::Vector3d nominal_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d nominal_moment = Eigen::Vector3d::Zero();
    for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }

        const int col = mahru::nDoF + DOF3 * contact;
        const Eigen::Vector3d r_CoM_to_contact =
            state_->fbk.p_C[contact] - robot_->p_CoM;

        a.block<DOF3, DOF3>(0, col).setIdentity();
        a.block<DOF3, DOF3>(DOF3, col) = Skew(r_CoM_to_contact);

        const Eigen::Vector3d nominal_grf =
            input_.grfs_mpc.segment<DOF3>(DOF3 * contact);
        nominal_force += nominal_grf;
        nominal_moment += r_CoM_to_contact.cross(nominal_grf);
    }

    if (!desired_force.allFinite()) {
        desired_force.setZero();
    }
    if (!desired_moment.allFinite()) {
        desired_moment.setZero();
    }

    b.segment<DOF3>(0) = desired_force - nominal_force;
    b.segment<DOF3>(DOF3) = desired_moment - nominal_moment;

    for (int i = 0; i < DOF6; ++i) {
        const double mask = std::clamp(input_.centroidal_force_task_mask(i), 0.0, 1.0);
        a.row(i) *= mask;
        b(i) *= mask;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateRollAngularMomentumTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    Eigen::Vector3d roll_axis = input_.roll_angular_momentum_axis;
    if (roll_axis.norm() < 1e-6 || !roll_axis.allFinite()) {
        roll_axis = Eigen::Vector3d::UnitX();
    } else {
        roll_axis.normalize();
    }

    a.block(0, 0, 1, mahru::nDoF) = roll_axis.transpose() * robot_->Ar_CoM;

    b(0) = input_.roll_angular_momentum_rate_d
        - (roll_axis.transpose() * robot_->Adotr_CoM * robot_->xidot)(0);

    if (!std::isfinite(b(0))) {
        b(0) = 0.0;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingLegRollMomentumTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    Eigen::Vector3d roll_axis = input_.roll_angular_momentum_axis;
    if (roll_axis.norm() < 1e-6 || !roll_axis.allFinite()) {
        roll_axis = Eigen::Vector3d::UnitX();
    } else {
        roll_axis.normalize();
    }

    std::array<int, 5> swing_leg_joints = {};
    if (input_.swing_contact_index == 1) {
        swing_leg_joints = {9, 10, 11, 12, 13};
    } else if (input_.swing_contact_index == 3) {
        swing_leg_joints = {14, 15, 16, 17, 18};
    } else {
        return WBCTask(kNumDecisionVars);
    }

    const Eigen::Matrix<double, 1, mahru::nDoF> cmm_row =
        roll_axis.transpose() * robot_->Ar_CoM;
    Eigen::VectorXd swing_xidot = Eigen::VectorXd::Zero(mahru::nDoF);
    for (const int joint : swing_leg_joints) {
        if (joint < 0 || joint >= static_cast<int>(mahru::num_act_joint)) {
            continue;
        }
        a(0, DOF6 + joint) = cmm_row(DOF6 + joint);
        swing_xidot(DOF6 + joint) = robot_->xidot(DOF6 + joint);
    }

    const double swing_bias =
        (roll_axis.transpose() * robot_->Adotr_CoM * swing_xidot)(0);
    b(0) = input_.swing_leg_roll_momentum_rate_d - swing_bias;
    if (!a.allFinite() || !std::isfinite(b(0))) {
        return WBCTask(kNumDecisionVars);
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateArmRollMomentumTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    Eigen::Vector3d roll_axis = input_.roll_angular_momentum_axis;
    if (roll_axis.norm() < 1e-6 || !roll_axis.allFinite()) {
        roll_axis = Eigen::Vector3d::UnitX();
    } else {
        roll_axis.normalize();
    }

    constexpr std::array<int, 2> shoulder_roll_joints = {2, 6};
    const Eigen::Matrix<double, 1, mahru::nDoF> cmm_row =
        roll_axis.transpose() * robot_->Ar_CoM;
    Eigen::VectorXd arm_xidot = Eigen::VectorXd::Zero(mahru::nDoF);
    for (const int joint : shoulder_roll_joints) {
        if (joint < 0 || joint >= static_cast<int>(mahru::num_act_joint)) {
            continue;
        }
        a(0, DOF6 + joint) = cmm_row(DOF6 + joint);
        arm_xidot(DOF6 + joint) = robot_->xidot(DOF6 + joint);
    }

    const double arm_bias =
        (roll_axis.transpose() * robot_->Adotr_CoM * arm_xidot)(0);
    b(0) = input_.arm_roll_momentum_rate_d - arm_bias;
    if (!a.allFinite() || !std::isfinite(b(0))) {
        return WBCTask(kNumDecisionVars);
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingLateralAccelerationTask()
{
    std::array<int, 2> swing_contacts = {
        input_.swing_contact_index,
        input_.secondary_swing_contact_index};
    int rows = 0;
    for (const int contact : swing_contacts) {
        if (contact >= 0 && contact < ConvexMpc::kNumContacts) {
            ++rows;
        }
    }
    if (rows == 0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::Vector3d lateral_axis = input_.swing_lateral_acceleration_axis;
    lateral_axis.z() = 0.0;
    if (lateral_axis.norm() < 1e-6 || !lateral_axis.allFinite()) {
        lateral_axis.setUnit(Y_AXIS);
    } else {
        lateral_axis.normalize();
    }

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    int row = 0;
    for (const int contact : swing_contacts) {
        if (contact < 0 || contact >= ConvexMpc::kNumContacts) {
            continue;
        }

        a.block(row, 0, 1, mahru::nDoF) =
            lateral_axis.transpose() * state_->fbk.Jp_C[contact];
        b(row) =
            input_.swing_lateral_acceleration_d
            - (lateral_axis.transpose()
               * state_->fbk.Jdotp_C[contact]
               * robot_->xidot)(0);
        if (!std::isfinite(b(row))) {
            b(row) = 0.0;
        }
        ++row;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateTorsoYawJointAccelerationTask()
{
    if (torso_yaw_joint_idx_ < 0
        || torso_yaw_joint_idx_ >= static_cast<int>(mahru::num_act_joint)) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    a(0, DOF6 + torso_yaw_joint_idx_) = 1.0;
    b(0) = kp_torso_yaw_jacc_ * (torso_yaw_d_ - robot_->q(torso_yaw_joint_idx_))
         + kd_torso_yaw_jacc_ * (-robot_->qdot(torso_yaw_joint_idx_));

    if (!std::isfinite(b(0))) {
        b(0) = 0.0;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingLegTask(
    const Eigen::Matrix3d& swingKp,
    const Eigen::Matrix3d& swingKd)
{
    return formulateSwingContactTask(
        input_.swing_contact_index,
        input_.p_sw_d,
        input_.pdot_sw_d,
        swingKp,
        swingKd);
}

WBCTask WeightedWBC::formulateSwingContactTask(
    int contact,
    const Eigen::Vector3d& p_d,
    const Eigen::Vector3d& pdot_d,
    const Eigen::Matrix3d& swingKp,
    const Eigen::Matrix3d& swingKd)
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF3, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF3);

    if (contact < 0 || contact >= ConvexMpc::kNumContacts) {
        return WBCTask(kNumDecisionVars);
    }

    const Eigen::Vector3d accel =
        swingKp * (p_d - state_->fbk.p_C[contact])
        + swingKd * (pdot_d - state_->fbk.pdot_C[contact]);

    a.block<DOF3, mahru::nDoF>(0, 0) = state_->fbk.Jp_C[contact];
    b = accel - state_->fbk.Jdotp_C[contact] * robot_->xidot;

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSlidingJointTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    int wheel_joint = -1;
    if (input_.swing_contact_index == 1) {
        wheel_joint = 18;
    } else if (input_.swing_contact_index == 3) {
        wheel_joint = 13;
    }

    if (wheel_joint >= 0) {
        constexpr double kWheelRadius = 0.079;
        constexpr double kSlidingKd = 40.0;
        constexpr double kMaxSlidingQddot = 120.0;
        const double desired_wheel_vel = input_.lin_vel_d(0) / kWheelRadius;
        const double desired_wheel_acc = input_.lin_acc_d(0) / kWheelRadius;
        const double vel_error = desired_wheel_vel - robot_->qdot(wheel_joint);
        const double qddot_cmd =
            std::clamp(
                desired_wheel_acc + kSlidingKd * vel_error,
                -kMaxSlidingQddot,
                kMaxSlidingQddot);

        a(0, DOF6 + wheel_joint) = 1.0;
        b(0) = qddot_cmd;
        diagnostics_.sliding_wheel_joint = wheel_joint;
        diagnostics_.sliding_vel_error = vel_error;
        diagnostics_.sliding_qddot_cmd = qddot_cmd;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingWheelTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    int wheel_joint = -1;
    if (input_.swing_contact_index == 1) {
        wheel_joint = 13;
    } else if (input_.swing_contact_index == 3) {
        wheel_joint = 18;
    }

    if (wheel_joint >= 0) {
        a(0, DOF6 + wheel_joint) = 1.0;
        diagnostics_.swing_wheel_joint = wheel_joint;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateJointAccelerationTask(
    const std::vector<int>& selectedJointsIdx,
    const Eigen::VectorXd& Kp,
    const Eigen::VectorXd& Kd)
{
    Eigen::MatrixXd a =
        Eigen::MatrixXd::Zero(selectedJointsIdx.size(), kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(a.rows());

    for (int i = 0; i < static_cast<int>(selectedJointsIdx.size()); ++i) {
        const int joint = selectedJointsIdx[i];
        if (joint < 0 || joint >= static_cast<int>(mahru::num_act_joint)) {
            continue;
        }

        a(i, DOF6 + joint) = 1.0;
        b(i) = input_.qddot_d(joint)
             + Kp(i) * (input_.q_d(joint) - robot_->q(joint))
             + Kd(i) * (input_.qdot_d(joint) - robot_->qdot(joint));
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateRegularizationTask(double qddot_regul, double rf_regul)
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Identity(kNumDecisionVars, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(kNumDecisionVars);

    a.block(0, 0, mahru::nDoF, mahru::nDoF) *= qddot_regul;
    a.block(mahru::nDoF, mahru::nDoF, ConvexMpc::kForceDim, ConvexMpc::kForceDim) *= rf_regul;

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateFloatingBaseConstraint()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF6, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF6);

    a.block(0, 0, DOF6, mahru::nDoF) =
        (Sf_ * robot_->M_mat).block(0, 0, DOF6, mahru::nDoF);
    a.block(0, mahru::nDoF, DOF6, ConvexMpc::kForceDim) =
        (-Sf_ * Jp_contact_.transpose()).block(0, 0, DOF6, ConvexMpc::kForceDim);

    b.noalias() =
        -1.0
        * (Sf_ * (robot_->C_mat * robot_->xidot
                  + robot_->g_vec
                  - Jp_contact_.transpose() * input_.grfs_mpc)).head<DOF6>();

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateContactNormalConstraint()
{
    int lateral_rows = 0;
    if (input_.enable_lateral_contact_constraint) {
        for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
            if (state_->ctrl.contact_schedule(contact, 0) == 0) {
                continue;
            }
            ++lateral_rows;
        }
    }
    int sagittal_rows = 0;
    if (input_.enable_sagittal_contact_constraint) {
        for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
            if (state_->ctrl.contact_schedule(contact, 0) == 0) {
                continue;
            }
            ++sagittal_rows;
        }
    }

    Eigen::MatrixXd a =
        Eigen::MatrixXd::Zero(
            numContactPoints_ + lateral_rows + sagittal_rows,
            kNumDecisionVars);
    Eigen::VectorXd b =
        Eigen::VectorXd::Zero(numContactPoints_ + lateral_rows + sagittal_rows);

    int row = 0;
    for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }

        a.block(row, 0, 1, mahru::nDoF) = state_->fbk.Jp_C[contact].row(2);
        b(row) = -(state_->fbk.Jdotp_C[contact].row(2) * robot_->xidot)(0);
        ++row;

        if (input_.enable_lateral_contact_constraint) {
            const Eigen::Vector3d lateral_axis = state_->fbk.R_C[contact].col(Y_AXIS);
            const Eigen::Matrix<double, 1, mahru::nDoF> J_lateral =
                lateral_axis.transpose() * state_->fbk.Jp_C[contact];
            const double Jdot_v_lateral =
                (lateral_axis.transpose() * state_->fbk.Jdotp_C[contact] * robot_->xidot)(0);
            const double lateral_vel = lateral_axis.dot(state_->fbk.pdot_C[contact]);

            a.block(row, 0, 1, mahru::nDoF) = J_lateral;
            b(row) = -Jdot_v_lateral
                - input_.lateral_contact_kd * lateral_vel;
            ++row;
        }

        if (input_.enable_sagittal_contact_constraint) {
            const Eigen::Vector3d sagittal_axis = state_->fbk.R_C[contact].col(X_AXIS);
            const Eigen::Matrix<double, 1, mahru::nDoF> J_sagittal =
                sagittal_axis.transpose() * state_->fbk.Jp_C[contact];
            const double Jdot_v_sagittal =
                (sagittal_axis.transpose()
                 * state_->fbk.Jdotp_C[contact]
                 * robot_->xidot)(0);
            const double sagittal_vel = sagittal_axis.dot(state_->fbk.pdot_C[contact]);

            a.block(row, 0, 1, mahru::nDoF) = J_sagittal;
            b(row) = -Jdot_v_sagittal
                - input_.sagittal_contact_kd * sagittal_vel;
            ++row;
        }
    }

    diagnostics_.contact_normal_rows = row;
    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingClearanceConstraint()
{
    if (!input_.enable_swing_clearance_constraint) {
        return WBCTask(kNumDecisionVars);
    }

    std::array<int, 2> swing_contacts = {
        input_.swing_contact_index,
        input_.secondary_swing_contact_index};
    int rows = 0;
    for (const int contact : swing_contacts) {
        if (contact >= 0 && contact < ConvexMpc::kNumContacts) {
            ++rows;
        }
    }
    if (rows == 0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(rows);

    int row = 0;
    for (const int contact : swing_contacts) {
        if (contact < 0 || contact >= ConvexMpc::kNumContacts) {
            continue;
        }

        const double z = state_->fbk.p_C[contact].z();
        const double zdot = state_->fbk.pdot_C[contact].z();
        const double jdot_v =
            (state_->fbk.Jdotp_C[contact].row(Z_AXIS) * robot_->xidot)(0);
        const double zddot_min = std::clamp(
            input_.swing_clearance_kp * (input_.swing_clearance_height - z)
            - input_.swing_clearance_kd * zdot,
            -input_.swing_clearance_max_acc,
            input_.swing_clearance_max_acc);

        d.block(row, 0, 1, mahru::nDoF) = state_->fbk.Jp_C[contact].row(Z_AXIS);
        f(row) = zddot_min - jdot_v;
        ++row;
    }

    return {Eigen::MatrixXd(), Eigen::VectorXd(), d, f};
}

WBCTask WeightedWBC::formulateSwingLateralClearanceConstraint()
{
    if (!input_.enable_swing_lateral_clearance_constraint) {
        return WBCTask(kNumDecisionVars);
    }

    std::array<int, 2> swing_contacts = {
        input_.swing_contact_index,
        input_.secondary_swing_contact_index};
    int rows = 0;
    for (const int contact : swing_contacts) {
        if (contact >= 0 && contact < ConvexMpc::kNumContacts) {
            ++rows;
        }
    }
    if (rows == 0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::Vector3d lateral_axis = input_.swing_lateral_clearance_axis;
    lateral_axis.z() = 0.0;
    if (lateral_axis.norm() < 1e-6 || !lateral_axis.allFinite()) {
        lateral_axis.setUnit(Y_AXIS);
    } else {
        lateral_axis.normalize();
    }
    const double side = input_.swing_lateral_clearance_side < 0.0 ? -1.0 : 1.0;

    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(rows);
    Eigen::Matrix<double, DOF3, mahru::nDoF> Jp_base =
        Eigen::Matrix<double, DOF3, mahru::nDoF>::Zero();
    Jp_base.block<DOF3, DOF3>(0, 0).setIdentity();

    int row = 0;
    for (const int contact : swing_contacts) {
        if (contact < 0 || contact >= ConvexMpc::kNumContacts) {
            continue;
        }

        const Eigen::Matrix<double, 1, mahru::nDoF> J_rel =
            lateral_axis.transpose() * (state_->fbk.Jp_C[contact] - Jp_base);
        const double distance =
            side * lateral_axis.dot(state_->fbk.p_C[contact] - state_->fbk.p_B);
        const double distance_dot =
            side * lateral_axis.dot(state_->fbk.pdot_C[contact] - state_->fbk.pdot_B);
        const double jdot_rel =
            side
            * (lateral_axis.transpose() * state_->fbk.Jdotp_C[contact]
               * robot_->xidot)(0);
        const double ddot_min = std::clamp(
            input_.swing_lateral_clearance_kp
                * (input_.swing_lateral_clearance_distance - distance)
            - input_.swing_lateral_clearance_kd * distance_dot,
            -input_.swing_lateral_clearance_max_acc,
            input_.swing_lateral_clearance_max_acc);

        d.block(row, 0, 1, mahru::nDoF) = side * J_rel;
        f(row) = ddot_min - jdot_rel;
        ++row;
    }

    return {Eigen::MatrixXd(), Eigen::VectorXd(), d, f};
}

WBCTask WeightedWBC::formulateFrictionConeConstraint()
{
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(ConvexMpc::kNumContacts * 5, kNumDecisionVars);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(ConvexMpc::kNumContacts * 5);

    Eigen::Matrix<double, 5, DOF3> W;
    W <<  1.0,  0.0, K_mu_,
         -1.0,  0.0, K_mu_,
          0.0,  1.0, K_mu_,
          0.0, -1.0, K_mu_,
          0.0,  0.0, 1.0;

    for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
        d.block<5, DOF3>(5 * contact, mahru::nDoF + DOF3 * contact) = W;
        f.segment<5>(5 * contact) =
            -W * input_.grfs_mpc.segment<DOF3>(DOF3 * contact);
    }

    return {Eigen::MatrixXd(), Eigen::VectorXd(), d, f};
}

WBCTask WeightedWBC::formulateAccelerationLimitConstraint()
{
    constexpr int kRows = 2 * mahru::num_act_joint;
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(kRows, kNumDecisionVars);
    Eigen::VectorXd f = Eigen::VectorXd::Constant(kRows, -joint_qddot_limit_);

    for (int joint = 0; joint < static_cast<int>(mahru::num_act_joint); ++joint) {
        d(2 * joint, DOF6 + joint) = 1.0;
        d(2 * joint + 1, DOF6 + joint) = -1.0;
    }

    return {Eigen::MatrixXd(), Eigen::VectorXd(), d, f};
}

bool WeightedWBC::solveQP(
    const WBCTask& constraints,
    const WBCTask& weightedTask,
    Eigen::VectorXd& qpSol) const
{
    const int numEqualities = static_cast<int>(constraints.b_.size());
    const int numInequalities = static_cast<int>(constraints.f_.size());
    const int numConstraints = numEqualities + numInequalities;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(numConstraints, kNumDecisionVars);
    Eigen::VectorXd lbA = Eigen::VectorXd::Zero(numConstraints);
    Eigen::VectorXd ubA = Eigen::VectorXd::Zero(numConstraints);

    if (numEqualities > 0) {
        A.topRows(numEqualities) = constraints.a_;
        lbA.head(numEqualities) = constraints.b_;
        ubA.head(numEqualities) = constraints.b_;
    }
    if (numInequalities > 0) {
        A.bottomRows(numInequalities) = constraints.d_;
        lbA.tail(numInequalities) = constraints.f_;
        ubA.tail(numInequalities).setConstant(1e20);
    }

    Eigen::MatrixXd H =
        weightedTask.a_.transpose() * weightedTask.a_
        + 1e-9 * Eigen::MatrixXd::Identity(kNumDecisionVars, kNumDecisionVars);
    Eigen::VectorXd g = -weightedTask.a_.transpose() * weightedTask.b_;

    Eigen::SparseMatrix<double> H_sparse = H.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A.sparseView();

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(false);
    solver.settings()->setMaxIteration(400);
    solver.settings()->setAbsoluteTolerance(1e-3);
    solver.settings()->setRelativeTolerance(1e-3);
    solver.data()->setNumberOfVariables(kNumDecisionVars);
    solver.data()->setNumberOfConstraints(numConstraints);

    if (!solver.data()->setHessianMatrix(H_sparse)
        || !solver.data()->setGradient(g)
        || !solver.data()->setLinearConstraintsMatrix(A_sparse)
        || !solver.data()->setLowerBound(lbA)
        || !solver.data()->setUpperBound(ubA)
        || !solver.initSolver()) {
        return false;
    }

    const auto exit_flag = solver.solveProblem();
    const auto status = solver.getStatus();
    if (exit_flag != OsqpEigen::ErrorExitFlag::NoError
        || (status != OsqpEigen::Status::Solved
            && status != OsqpEigen::Status::SolvedInaccurate)) {
        return false;
    }

    qpSol = solver.getSolution();
    return qpSol.allFinite();
}

void WeightedWBC::loadWeightGain()
{
    const std::string path = CMAKE_SOURCE_DIR "/config/config_weighted.yaml";
    YAML::Node yaml_node = YAML::LoadFile(path.c_str());

    try {
        if (yaml_node["W_swingLeg"]) {
            W_swingLeg_ = yaml_node["W_swingLeg"].as<double>();
        }
        W_JointAcc_ = yaml_node["W_JointAcc"].as<double>();
        W_qddot_regul_ = yaml_node["W_qddot_regul"].as<double>();
        W_rf_regul_ = yaml_node["W_rf_regul"].as<double>();
        W_centroidal_ = yaml_node["W_Centroidal"].as<double>();
        if (yaml_node["W_CenAngMom_Compen"]) {
            W_CenAngMom_Compen_ = yaml_node["W_CenAngMom_Compen"].as<double>();
        }
        if (yaml_node["W_CentroidalForce"]) {
            W_centroidal_force_ = yaml_node["W_CentroidalForce"].as<double>();
        }
        if (yaml_node["W_RollAngularMomentum"]) {
            W_roll_angular_momentum_ = yaml_node["W_RollAngularMomentum"].as<double>();
        }
        if (yaml_node["W_SwingLegRollMomentum"]) {
            W_swing_leg_roll_momentum_ =
                yaml_node["W_SwingLegRollMomentum"].as<double>();
        }
        if (yaml_node["W_ArmRollMomentum"]) {
            W_arm_roll_momentum_ = yaml_node["W_ArmRollMomentum"].as<double>();
        }
        if (yaml_node["W_SwingLateralAccel"]) {
            W_swing_lateral_acceleration_ =
                yaml_node["W_SwingLateralAccel"].as<double>();
        }
        if (yaml_node["W_TorsoYawJointAcc"]) {
            W_torso_yaw_joint_acc_ = yaml_node["W_TorsoYawJointAcc"].as<double>();
        }
        if (yaml_node["W_wheelAccel"]) {
            W_wheelAccel_ = yaml_node["W_wheelAccel"].as<double>();
        }
        if (yaml_node["JointAccLimit"] && yaml_node["JointAccLimit"]["qddot"]) {
            joint_qddot_limit_ = yaml_node["JointAccLimit"]["qddot"].as<double>();
        }

        kp_CoM_.setZero();
        kd_CoM_.setZero();
        kp_R_.setZero();
        kd_omega_.setZero();
        kp_swing_.setZero();
        kd_swing_.setZero();
        for (int i = 0; i < static_cast<int>(DOF3); ++i) {
            kp_CoM_(i, i) = yaml_node["CentroidalTask"]["com_kp"][i].as<double>();
            kd_CoM_(i, i) = yaml_node["CentroidalTask"]["com_kd"][i].as<double>();
            kp_R_(i, i) = yaml_node["CentroidalTask"]["orientation_kp"][i].as<double>();
            kd_omega_(i, i) = yaml_node["CentroidalTask"]["orientation_kd"][i].as<double>();

            if (yaml_node["SwingTask"]) {
                kp_swing_(i, i) = yaml_node["SwingTask"]["kp"][i].as<double>();
                kd_swing_(i, i) = yaml_node["SwingTask"]["kd"][i].as<double>();
            } else {
                kp_swing_(i, i) = 1000.0;
                kd_swing_(i, i) = 40.0;
            }
        }

        selectedJointsIdx_.clear();
        kp_jacc_ = Eigen::VectorXd::Zero(yaml_node["JointAccTask"]["joints"].size());
        kd_jacc_ = Eigen::VectorXd::Zero(yaml_node["JointAccTask"]["joints"].size());
        for (int i = 0; i < static_cast<int>(yaml_node["JointAccTask"]["joints"].size()); ++i) {
            selectedJointsIdx_.push_back(yaml_node["JointAccTask"]["joints"][i].as<int>());
            kp_jacc_(i) = yaml_node["JointAccTask"]["kp"][i].as<double>();
            kd_jacc_(i) = yaml_node["JointAccTask"]["kd"][i].as<double>();
        }

        if (yaml_node["TorsoYawJointAccTask"]) {
            const YAML::Node torso_yaw = yaml_node["TorsoYawJointAccTask"];
            if (torso_yaw["joint"]) {
                torso_yaw_joint_idx_ = torso_yaw["joint"].as<int>();
            }
            if (torso_yaw["q_d_deg"]) {
                torso_yaw_d_ = torso_yaw["q_d_deg"].as<double>() * M_PI / 180.0;
            } else if (torso_yaw["q_d"]) {
                torso_yaw_d_ = torso_yaw["q_d"].as<double>();
            }
            if (torso_yaw["kp"]) {
                kp_torso_yaw_jacc_ = torso_yaw["kp"].as<double>();
            }
            if (torso_yaw["kd"]) {
                kd_torso_yaw_jacc_ = torso_yaw["kd"].as<double>();
            }
        }

        K_f_ = yaml_node["ContactWrench"]["k_f"].as<double>();
        if (yaml_node["ContactWrench"]["k_mu"]) {
            K_mu_ = yaml_node["ContactWrench"]["k_mu"].as<double>();
        }

        std::cout << "WeightedWBC Weight Gain Config YAML File Loaded!" << std::endl;
    } catch (const std::exception&) {
        std::cerr << "Fail to read WeightedWBC Weight Gain Config YAML file" << std::endl;
    }
}
