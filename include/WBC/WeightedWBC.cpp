#include "WBC/WeightedWBC.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <qpOASES.hpp>

namespace {
double planarYaw(const Eigen::Matrix3d& rotation)
{
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

double wrapToPi(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}
} // namespace

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
    if (input_.swing_contact_index == 1 || input_.swing_contact_index == 3) {
        numContactPoints_ = 1;
    } else {
        numContactPoints_ = 4;
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
        + formulateFrictionConeConstraint();
}

WBCTask WeightedWBC::formulateWeightedTask()
{
    if (numContactPoints_ == 4 || numContactPoints_ == 2) {
        return formulateJointAccelerationTask(selectedJointsIdx_, kp_jacc_, kd_jacc_)
                   * W_JointAcc_
             + formulateSlidingJointTask()
             + formulateRegularizationTask(W_qddot_regul_, W_rf_regul_);
    }

    if (numContactPoints_ == 1) {
        return formulateLinearMotionTask() * W_centroidal_
             + formulateJointAccelerationTask(selectedJointsIdx_, kp_jacc_, kd_jacc_)
                   * W_JointAcc_
             + formulateSwingLegTask(kp_swing_, kd_swing_) * W_swingLeg_
             + formulateSlidingJointTask()
             + formulateRegularizationTask(W_qddot_regul_, W_rf_regul_);
    }

    return formulateJointAccelerationTask(selectedJointsIdx_, kp_jacc_, kd_jacc_)
               * W_JointAcc_
         + formulateSlidingJointTask()
         + formulateRegularizationTask(W_qddot_regul_, W_rf_regul_);
}

WBCTask WeightedWBC::formulateLinearMotionTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF3, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF3);

    const Eigen::Matrix3d errorMatrix =
        Eigen::Matrix3d::Identity() * robot_->R_B.transpose();
    const Eigen::AngleAxisd errorAngleAxis(errorMatrix);
    Eigen::Vector3d oriErr = errorAngleAxis.angle() * errorAngleAxis.axis();
    if (!oriErr.allFinite()) {
        oriErr.setZero();
    }
    Eigen::Vector3d omega_d = Eigen::Vector3d::Zero();

    a(0, 2) = 1.0;
    a(1, 3) = 1.0;
    a(2, 4) = 1.0;
    b(0) = 50.0 * (0.7 - robot_->p_CoM(2))
         + 2.0 * (0.0 - robot_->pdot_CoM(2));
    b.segment<2>(1) =
        (500.0 * oriErr + 40.0 * (omega_d - robot_->omega_B)).segment<2>(0);

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulatePelvisYawTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    const double yaw_d = planarYaw(input_.R_B_d);
    const double yaw = planarYaw(robot_->R_B);
    const double yaw_error = wrapToPi(yaw_d - yaw);
    const double yaw_rate_error = input_.ang_vel_d.z() - robot_->omega_B.z();

    double yaw_acc_cmd =
        pelvis_yaw_kp_ * yaw_error
        + pelvis_yaw_kd_ * yaw_rate_error;
    if (pelvis_yaw_acc_limit_ > 0.0) {
        yaw_acc_cmd =
            std::clamp(yaw_acc_cmd,
                       -pelvis_yaw_acc_limit_,
                       pelvis_yaw_acc_limit_);
    }

    a(0, 5) = 1.0;
    b(0) = yaw_acc_cmd;

    if (!std::isfinite(b(0))) {
        return WBCTask(kNumDecisionVars);
    }

    diagnostics_.pelvis_yaw_error = yaw_error;
    diagnostics_.pelvis_yaw_acc_cmd = yaw_acc_cmd;
    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateUpperBodyMomentumCompensationTask()
{
    if (input_.swing_contact_index != 1 && input_.swing_contact_index != 3) {
        return WBCTask(kNumDecisionVars);
    }

    const std::array<int, 5> swing_leg_joints =
        input_.swing_contact_index == 1
            ? std::array<int, 5>{9, 10, 11, 12, 13}
            : std::array<int, 5>{14, 15, 16, 17, 18};

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF3, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF3);

    Eigen::Vector3d h_upper = Eigen::Vector3d::Zero();
    Eigen::Vector3d h_swing = Eigen::Vector3d::Zero();
    Eigen::Vector3d hdot_upper_bias = Eigen::Vector3d::Zero();

    for (const int joint : upperBodyMomentumJoints_) {
        if (joint < 0 || joint >= static_cast<int>(mahru::num_act_joint)) {
            continue;
        }
        const int dof = DOF6 + joint;
        a.block<DOF3, 1>(0, dof) = robot_->Ar_CoM.col(dof);
        h_upper.noalias() += robot_->Ar_CoM.col(dof) * robot_->xidot(dof);
        hdot_upper_bias.noalias() += robot_->Adotr_CoM.col(dof) * robot_->xidot(dof);
    }

    for (const int joint : swing_leg_joints) {
        const int dof = DOF6 + joint;
        h_swing.noalias() += robot_->Ar_CoM.col(dof) * robot_->xidot(dof);
    }

    Eigen::Vector3d mask = input_.upper_body_momentum_axis_mask;
    for (int i = 0; i < DOF3; ++i) {
        mask(i) = std::clamp(mask(i), 0.0, 1.0);
    }
    if (mask.norm() < 1e-6) {
        return WBCTask(kNumDecisionVars);
    }

    b = -hdot_upper_bias
        - input_.upper_body_momentum_kp * (h_upper + h_swing);

    if (input_.upper_body_momentum_max_rate > 0.0) {
        for (int i = 0; i < DOF3; ++i) {
            b(i) = std::clamp(
                b(i),
                -input_.upper_body_momentum_max_rate,
                input_.upper_body_momentum_max_rate);
        }
    }

    for (int row = 0; row < DOF3; ++row) {
        a.row(row) *= mask(row);
        b(row) *= mask(row);
    }

    if (!a.allFinite() || !b.allFinite()) {
        return WBCTask(kNumDecisionVars);
    }

    diagnostics_.swing_leg_angular_momentum_norm = h_swing.norm();
    diagnostics_.upper_body_angular_momentum_norm = h_upper.norm();
    diagnostics_.upper_body_momentum_task_max = b.cwiseAbs().maxCoeff();

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateUpperBodyPostureTask()
{
    Eigen::MatrixXd a =
        Eigen::MatrixXd::Zero(upperBodyMomentumJoints_.size(), kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(a.rows());

    for (int i = 0; i < static_cast<int>(upperBodyMomentumJoints_.size()); ++i) {
        const int joint = upperBodyMomentumJoints_[i];
        if (joint < 0 || joint >= static_cast<int>(mahru::num_act_joint)) {
            continue;
        }

        const int dof = DOF6 + joint;
        a(i, dof) = 1.0;
        b(i) = input_.qddot_d(joint)
             + input_.upper_body_posture_kp
               * (input_.q_d(joint) - robot_->q(joint))
             + input_.upper_body_posture_kd
               * (input_.qdot_d(joint) - robot_->qdot(joint));

        if (joint_qddot_limit_ > 0.0) {
            b(i) = std::clamp(b(i), -joint_qddot_limit_, joint_qddot_limit_);
        }
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSwingLegTask(
    const Eigen::Matrix3d& swingKp,
    const Eigen::Matrix3d& swingKd)
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(DOF3, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(DOF3);

    const Eigen::Vector3d accel =
        swingKp * (input_.p_sw_d - p_sw_)
        + swingKd * (input_.pdot_sw_d - pdot_sw_);

    a.block<DOF3, mahru::nDoF>(0, 0) = Jp_sw_;
    b = accel - Jdotp_sw_ * robot_->xidot;

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSlidingJointTask()
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(1, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

    int wheel_joint = -1;
    if (input_.previous_state_machine == 1) {
        wheel_joint = 18;
    } else if (input_.previous_state_machine == 2) {
        wheel_joint = 13;
    }

    if (wheel_joint >= 0) {
        constexpr double kWheelRadius = 0.079;
        constexpr double kSlidingKd = 400.0;
        const double desired_wheel_vel = input_.lin_vel_d(0) / kWheelRadius;
        const double vel_error = desired_wheel_vel - robot_->qdot(wheel_joint);

        a(0, DOF6 + wheel_joint) = 1.0;
        b(0) = kSlidingKd * vel_error;
        diagnostics_.sliding_wheel_joint = wheel_joint;
        diagnostics_.sliding_vel_error = vel_error;
        diagnostics_.sliding_qddot_cmd = b(0);
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSlidingContactTask()
{
    constexpr std::array<int, 2> kWheelContacts = {1, 3};

    Eigen::Vector3d mask = sliding_contact_axis_mask_;
    for (int i = 0; i < DOF3; ++i) {
        mask(i) = std::clamp(mask(i), 0.0, 1.0);
    }

    int rows = 0;
    for (const int contact : kWheelContacts) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }
        for (int axis = 0; axis < DOF3; ++axis) {
            if (mask(axis) > 1e-6) {
                ++rows;
            }
        }
    }

    if (rows == 0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    int row = 0;
    double error_norm = 0.0;
    for (const int contact : kWheelContacts) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }

        const Eigen::Vector3d accel =
            sliding_contact_kp_
                * (input_.p_slide_d.col(contact) - state_->fbk.p_C[contact])
            + sliding_contact_kd_
                * (input_.pdot_slide_d.col(contact) - state_->fbk.pdot_C[contact]);
        error_norm +=
            (input_.p_slide_d.col(contact) - state_->fbk.p_C[contact]).norm();

        for (int axis = 0; axis < DOF3; ++axis) {
            if (mask(axis) <= 1e-6) {
                continue;
            }
            a.block(row, 0, 1, mahru::nDoF) =
                mask(axis) * state_->fbk.Jp_C[contact].row(axis);
            double axis_acc =
                mask(axis)
                * (accel(axis)
                   - (state_->fbk.Jdotp_C[contact].row(axis) * robot_->xidot)(0));
            if (sliding_contact_acc_limit_ > 0.0) {
                axis_acc = std::clamp(axis_acc,
                                      -sliding_contact_acc_limit_,
                                      sliding_contact_acc_limit_);
            }
            b(row) = axis_acc;
            ++row;
        }
    }

    diagnostics_.sliding_contact_error_norm = error_norm;
    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateSupportWheelTask()
{
    constexpr std::array<int, 2> kWheelContacts = {1, 3};
    constexpr std::array<int, 2> kWheelJoints = {13, 18};

    int rows = 0;
    for (const int contact : kWheelContacts) {
        if (state_->ctrl.contact_schedule(contact, 0) != 0) {
            ++rows;
        }
    }

    if (rows == 0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    constexpr double kWheelRadius = 0.079;
    constexpr double kWheelKd = 40.0;
    constexpr double kMaxWheelQddot = 120.0;
    const double desired_wheel_vel = input_.lin_vel_d(0) / kWheelRadius;

    int row = 0;
    for (int i = 0; i < static_cast<int>(kWheelContacts.size()); ++i) {
        const int contact = kWheelContacts[i];
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }
        const int wheel_joint = kWheelJoints[i];
        const double vel_error = desired_wheel_vel - robot_->qdot(wheel_joint);
        const double qddot_cmd =
            std::clamp(kWheelKd * vel_error, -kMaxWheelQddot, kMaxWheelQddot);

        a(row, DOF6 + wheel_joint) = 1.0;
        b(row) = qddot_cmd;
        diagnostics_.sliding_wheel_joint = wheel_joint;
        diagnostics_.sliding_vel_error = vel_error;
        diagnostics_.sliding_qddot_cmd = qddot_cmd;
        ++row;
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
        constexpr double kWheelRadius = 0.079;
        constexpr double kWheelKd = 40.0;
        constexpr double kMaxWheelQddot = 120.0;
        const double desired_wheel_vel = input_.lin_vel_d(0) / kWheelRadius;
        const double vel_error = desired_wheel_vel - robot_->qdot(wheel_joint);

        a(0, DOF6 + wheel_joint) = 1.0;
        b(0) =
            std::clamp(kWheelKd * vel_error, -kMaxWheelQddot, kMaxWheelQddot);
        diagnostics_.swing_wheel_joint = wheel_joint;
    }

    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
}

WBCTask WeightedWBC::formulateRollingContactTask()
{
    constexpr std::array<int, 2> kWheelContacts = {1, 3};

    int rows = 0;
    for (const int contact : kWheelContacts) {
        if (state_->ctrl.contact_schedule(contact, 0) != 0) {
            ++rows;
        }
    }
    if (rows == 0 || W_rolling_contact_ <= 0.0) {
        return WBCTask(kNumDecisionVars);
    }

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(rows, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

    int row = 0;
    double vel_error_max = 0.0;
    double acc_cmd_max = 0.0;
    for (const int contact : kWheelContacts) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }

        Eigen::Vector3d tangent = state_->fbk.R_C[contact].col(0);
        if (!tangent.allFinite() || tangent.norm() < 1e-6) {
            continue;
        }
        tangent.normalize();

        const double tangential_velocity =
            tangent.dot(state_->fbk.pdot_C[contact]);
        const double jacobian_bias =
            (tangent.transpose()
             * state_->fbk.Jdotp_C[contact]
             * robot_->xidot)(0);
        double tangential_acc_cmd =
            -rolling_contact_kd_ * tangential_velocity;
        if (rolling_contact_acc_limit_ > 0.0) {
            tangential_acc_cmd =
                std::clamp(tangential_acc_cmd,
                           -rolling_contact_acc_limit_,
                           rolling_contact_acc_limit_);
        }

        a.block(row, 0, 1, mahru::nDoF) =
            tangent.transpose() * state_->fbk.Jp_C[contact];
        b(row) = tangential_acc_cmd - jacobian_bias;

        vel_error_max = std::max(vel_error_max, std::abs(tangential_velocity));
        acc_cmd_max = std::max(acc_cmd_max, std::abs(tangential_acc_cmd));
        ++row;
    }

    diagnostics_.rolling_contact_vel_error = vel_error_max;
    diagnostics_.rolling_contact_acc_cmd = acc_cmd_max;
    return {a.topRows(row), b.head(row), Eigen::MatrixXd(), Eigen::VectorXd()};
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
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(numContactPoints_, kNumDecisionVars);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(numContactPoints_);

    int row = 0;
    for (int contact = 0; contact < ConvexMpc::kNumContacts; ++contact) {
        if (state_->ctrl.contact_schedule(contact, 0) == 0) {
            continue;
        }

        a.block(row, 0, 1, mahru::nDoF) = state_->fbk.Jp_C[contact].row(2);
        b(row) = -(state_->fbk.Jdotp_C[contact].row(2) * robot_->xidot)(0);
        ++row;
    }

    diagnostics_.contact_normal_rows = row;
    return {a, b, Eigen::MatrixXd(), Eigen::VectorXd()};
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
    const int numConstraints =
        static_cast<int>(constraints.b_.size() + constraints.f_.size());
    using RowMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    RowMajorMatrix A(numConstraints, kNumDecisionVars);
    Eigen::VectorXd lbA(numConstraints);
    Eigen::VectorXd ubA(numConstraints);
    A << constraints.a_, constraints.d_;
    lbA << constraints.b_, constraints.f_;
    ubA << constraints.b_,
        qpOASES::INFTY * Eigen::VectorXd::Ones(constraints.f_.size());

    RowMajorMatrix H =
        weightedTask.a_.transpose() * weightedTask.a_;
    Eigen::VectorXd g = -weightedTask.a_.transpose() * weightedTask.b_;

    qpOASES::QProblem qpProblem(kNumDecisionVars, numConstraints);
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_LOW;
    options.enableEqualities = qpOASES::BT_TRUE;
    qpProblem.setOptions(options);

    int nWsr = 50;
    const qpOASES::returnValue init_status =
        qpProblem.init(H.data(),
                       g.data(),
                       A.data(),
                       nullptr,
                       nullptr,
                       lbA.data(),
                       ubA.data(),
                       nWsr);
    if (init_status != qpOASES::SUCCESSFUL_RETURN) {
        return false;
    }

    qpSol = Eigen::VectorXd::Zero(kNumDecisionVars);
    const qpOASES::returnValue solution_status =
        qpProblem.getPrimalSolution(qpSol.data());
    return solution_status == qpOASES::SUCCESSFUL_RETURN && qpSol.allFinite();
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
        if (yaml_node["W_wheelAccel"]) {
            W_wheelAccel_ = yaml_node["W_wheelAccel"].as<double>();
        }
        if (yaml_node["W_SlidingContact"]) {
            W_sliding_contact_ = yaml_node["W_SlidingContact"].as<double>();
        }
        if (yaml_node["W_RollingContact"]) {
            W_rolling_contact_ = yaml_node["W_RollingContact"].as<double>();
        }
        if (yaml_node["W_PelvisYaw"]) {
            W_pelvis_yaw_ = yaml_node["W_PelvisYaw"].as<double>();
        }
        if (yaml_node["W_UpperBodyMomentumCompensation"]) {
            W_upper_body_momentum_compensation_ =
                yaml_node["W_UpperBodyMomentumCompensation"].as<double>();
        }
        if (yaml_node["W_UpperBodyPosture"]) {
            W_upper_body_posture_ =
                yaml_node["W_UpperBodyPosture"].as<double>();
        }
        if (yaml_node["JointAccLimit"] && yaml_node["JointAccLimit"]["qddot"]) {
            joint_qddot_limit_ = yaml_node["JointAccLimit"]["qddot"].as<double>();
        }
        if (yaml_node["SlidingContactTask"]) {
            const YAML::Node sliding_node = yaml_node["SlidingContactTask"];
            if (sliding_node["axis_mask"]
                && sliding_node["axis_mask"].IsSequence()
                && sliding_node["axis_mask"].size() >= DOF3) {
                for (int i = 0; i < DOF3; ++i) {
                    sliding_contact_axis_mask_(i) =
                        sliding_node["axis_mask"][i].as<double>();
                }
            }
            if (sliding_node["kp"]) {
                sliding_contact_kp_ = sliding_node["kp"].as<double>();
            }
            if (sliding_node["kd"]) {
                sliding_contact_kd_ = sliding_node["kd"].as<double>();
            }
            if (sliding_node["acc_limit"]) {
                sliding_contact_acc_limit_ =
                    sliding_node["acc_limit"].as<double>();
            }
        }
        if (yaml_node["RollingContactTask"]) {
            const YAML::Node rolling_node = yaml_node["RollingContactTask"];
            if (rolling_node["kd"]) {
                rolling_contact_kd_ = rolling_node["kd"].as<double>();
            }
            if (rolling_node["acc_limit"]) {
                rolling_contact_acc_limit_ =
                    rolling_node["acc_limit"].as<double>();
            }
        }
        if (yaml_node["PelvisYawTask"]) {
            const YAML::Node yaw_node = yaml_node["PelvisYawTask"];
            if (yaw_node["kp"]) {
                pelvis_yaw_kp_ = yaw_node["kp"].as<double>();
            }
            if (yaw_node["kd"]) {
                pelvis_yaw_kd_ = yaw_node["kd"].as<double>();
            }
            if (yaw_node["acc_limit"]) {
                pelvis_yaw_acc_limit_ = yaw_node["acc_limit"].as<double>();
            }
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

        upperBodyMomentumJoints_.clear();
        if (yaml_node["UpperBodyMomentumCompensation"]
            && yaml_node["UpperBodyMomentumCompensation"]["upper_joints"]) {
            const YAML::Node upper_joints =
                yaml_node["UpperBodyMomentumCompensation"]["upper_joints"];
            for (int i = 0; i < static_cast<int>(upper_joints.size()); ++i) {
                upperBodyMomentumJoints_.push_back(upper_joints[i].as<int>());
            }
        } else {
            upperBodyMomentumJoints_ = {0, 1, 2, 3, 4, 5, 6, 7, 8};
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
