#include "WBC/WeightedWBC.hpp"

#include <algorithm>
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
        + formulateFrictionConeConstraint()
        + formulateAccelerationLimitConstraint();
}

WBCTask WeightedWBC::formulateWeightedTask()
{
    WBCTask task(kNumDecisionVars);

    task = task
        + formulateJointAccelerationTask(selectedJointsIdx_, kp_jacc_, kd_jacc_) * W_JointAcc_
        + formulateSlidingJointTask()
        + formulateLinearMotionTask() * W_centroidal_;

    if (input_.enable_centroidal_force_task) {
        task = task + formulateCentroidalForceTask() * W_centroidal_force_;
    }

    if (input_.enable_roll_angular_momentum_task) {
        task = task + formulateRollAngularMomentumTask() * W_roll_angular_momentum_;
    }

    if (W_torso_yaw_joint_acc_ > 0.0) {
        task = task + formulateTorsoYawJointAccelerationTask() * W_torso_yaw_joint_acc_;
    }

    if (input_.swing_contact_index >= 0 && numContactPoints_ < ConvexMpc::kNumContacts) {
        task = task
            + formulateSwingLegTask(kp_swing_, kd_swing_) * W_swingLeg_
            + formulateSwingWheelTask() * W_wheelAccel_;
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

    // Do not let torso yaw or arms generate this roll-momentum objective.
    for (int joint = 0; joint <= 8; ++joint) {
        a(0, DOF6 + joint) = 0.0;
    }

    b(0) = input_.roll_angular_momentum_rate_d
        - (roll_axis.transpose() * robot_->Adotr_CoM * robot_->xidot)(0);

    if (!std::isfinite(b(0))) {
        b(0) = 0.0;
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
        const double vel_error = desired_wheel_vel - robot_->qdot(wheel_joint);
        const double qddot_cmd =
            std::clamp(kSlidingKd * vel_error, -kMaxSlidingQddot, kMaxSlidingQddot);

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

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
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
