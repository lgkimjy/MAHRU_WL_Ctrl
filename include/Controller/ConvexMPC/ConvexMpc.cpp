#if 0
//
// Created by zixin on 12/09/21.
//

#include "ConvexMpc.h"

ConvexMpc::ConvexMpc(Eigen::VectorXd &q_weights_, Eigen::VectorXd &r_weights_) 
{
    mu = 0.7;
    fz_min = 0.0;
    fz_max = 0.0;

    // reserve size for sparse matrix
    Q_sparse = Eigen::SparseMatrix<double>(MPC_STATE_DIM * PLAN_HORIZON,MPC_STATE_DIM * PLAN_HORIZON);
    R_sparse = Eigen::SparseMatrix<double>(NUM_DOF * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON);

    q_weights_mpc.resize(MPC_STATE_DIM * PLAN_HORIZON);
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        q_weights_mpc.segment(i * MPC_STATE_DIM, MPC_STATE_DIM) = q_weights_;
    }
    Q.diagonal() = 2*q_weights_mpc;
    for (int i = 0; i < MPC_STATE_DIM*PLAN_HORIZON; ++i) {
        Q_sparse.insert(i,i) = 2*q_weights_mpc(i);
    }

//    Q.setZero();
//    R.setZero();
//
//    Eigen::Matrix<double, MPC_STATE_DIM, MPC_STATE_DIM> Q_small;
//    Q_small.setZero();
//    for (int i = 0; i < MPC_STATE_DIM; ++i) {
//        Q_small(i, i) = q_weights_[i];
//    }
//    for (int i = 0; i < PLAN_HORIZON; ++i) {
//        Q.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * i, MPC_STATE_DIM * i) = Q_small;
//    }

    r_weights_mpc.resize(NUM_DOF * PLAN_HORIZON);
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        r_weights_mpc.segment(i * NUM_DOF, NUM_DOF) = r_weights_;
    }
    R.diagonal() = 2*r_weights_mpc;
    for (int i = 0; i < NUM_DOF*PLAN_HORIZON; ++i) {
        R_sparse.insert(i,i) = 2*r_weights_mpc(i);
    }

    linear_constraints.resize(MPC_CONSTRAINT_DIM * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON);    
    for (int i = 0; i < NUM_LEG * PLAN_HORIZON; ++i) {
        linear_constraints.insert(0 + 5 * i, 0 + 3 * i) = 1;
        linear_constraints.insert(1 + 5 * i, 0 + 3 * i) = 1;
        linear_constraints.insert(2 + 5 * i, 1 + 3 * i) = 1;
        linear_constraints.insert(3 + 5 * i, 1 + 3 * i) = 1;
        linear_constraints.insert(4 + 5 * i, 2 + 3 * i) = 1;

        linear_constraints.insert(0 + 5 * i, 2 + 3 * i) = mu;
        linear_constraints.insert(1 + 5 * i, 2 + 3 * i) = -mu;
        linear_constraints.insert(2 + 5 * i, 2 + 3 * i) = mu;
        linear_constraints.insert(3 + 5 * i, 2 + 3 * i) = -mu;
    }

//    Eigen::Matrix<double, NUM_DOF, NUM_DOF> R_small;
//    R_small.setZero();
//    for (int i = 0; i < NUM_DOF; ++i) {
//        R_small(i, i) = r_weights_[i];
//    }
//    for (int i = 0; i < PLAN_HORIZON; ++i) {
//        R.block<NUM_DOF, NUM_DOF>(NUM_DOF * i, NUM_DOF * i) = R_small;
//    }
}
#endif

#include "Controller/ConvexMPC/ConvexMpc.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {
Eigen::Matrix3d skew(const Eigen::Vector3d& vec)
{
    Eigen::Matrix3d mat;
    mat << 0.0, -vec.z(), vec.y(),
           vec.z(), 0.0, -vec.x(),
           -vec.y(), vec.x(), 0.0;
    return mat;
}

template <typename MatrixType>
Eigen::SparseMatrix<double> sparseFromDense(const MatrixType& dense)
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(dense.rows() * dense.cols()));
    for (int row = 0; row < dense.rows(); ++row) {
        for (int col = 0; col < dense.cols(); ++col) {
            triplets.emplace_back(row, col, dense(row, col));
        }
    }

    Eigen::SparseMatrix<double> sparse(dense.rows(), dense.cols());
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();
    return sparse;
}
}

ConvexMpc::ConvexMpc()
{
    q_weights_ << 100.0, 100.0, 1000.0,
                  10.0, 10.0, 2700.0,
                  20.0, 20.0, 200.0,
                  20.0, 20.0, 20.0,
                  0.0;

    r_weights_ << 1e-5, 1e-5, 1e-6,
                  1e-5, 1e-5, 1e-6,
                  1e-5, 1e-5, 1e-6,
                  1e-5, 1e-5, 1e-6;

    for (int i = 0; i < kPlanHorizon; ++i) {
        Q_.diagonal().segment<kStateDim>(i * kStateDim) = 2.0 * q_weights_;
        R_.diagonal().segment<kForceDim>(i * kForceDim) = 2.0 * r_weights_;
    }

    constraint_mat_.resize(kNumConstraints, kNumDecisionVars);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(kNumConstraints * 3);
    for (int i = 0; i < kNumContacts * kPlanHorizon; ++i) {
        triplets.emplace_back(0 + 5 * i, 0 + 3 * i, 1.0);
        triplets.emplace_back(1 + 5 * i, 0 + 3 * i, 1.0);
        triplets.emplace_back(2 + 5 * i, 1 + 3 * i, 1.0);
        triplets.emplace_back(3 + 5 * i, 1 + 3 * i, 1.0);
        triplets.emplace_back(4 + 5 * i, 2 + 3 * i, 1.0);

        triplets.emplace_back(0 + 5 * i, 2 + 3 * i, mu_);
        triplets.emplace_back(1 + 5 * i, 2 + 3 * i, -mu_);
        triplets.emplace_back(2 + 5 * i, 2 + 3 * i, mu_);
        triplets.emplace_back(3 + 5 * i, 2 + 3 * i, -mu_);
    }
    constraint_mat_.setFromTriplets(triplets.begin(), triplets.end());
    constraint_mat_.makeCompressed();

    reset();
}

void ConvexMpc::reset()
{
    A_mat_c_.setZero();
    B_mat_c_.setZero();
    B_mat_d_list_.setZero();
    A_mat_d_.setZero();
    B_mat_d_.setZero();
    A_qp_.setZero();
    B_qp_.setZero();
    lb_.setZero();
    ub_.setZero();
    grf_.setZero();
    predicted_states_.setZero();
    gradient_osqp_.setZero(kNumDecisionVars);
    lb_osqp_.setZero(kNumConstraints);
    ub_osqp_.setZero(kNumConstraints);

    if (solver_.isInitialized()) {
        solver_.clearSolver();
    }
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();
    solver_initialized_ = false;
}

bool ConvexMpc::update(const Input& input, double dt)
{
    if (input.mass <= 0.0 || dt <= 0.0) {
        return false;
    }

    Eigen::Matrix<double, kStateDim, 1> mpc_states;
    mpc_states << input.euler_B,
                  input.p_CoM,
                  input.omega_B,
                  input.pdot_CoM,
                  -9.8;

    const Eigen::Vector3d lin_vel_d_world = input.R_B * input.lin_vel_d;
    const Eigen::Vector3d ang_vel_d_world = input.R_B * input.ang_vel_d;

    Eigen::Matrix<double, kStateDim * kPlanHorizon, 1> mpc_states_d;
    for (int i = 0; i < kPlanHorizon; ++i) {
        const double t = dt * static_cast<double>(i);
        mpc_states_d.segment<kStateDim>(i * kStateDim) <<
            input.euler_B_d,
            input.p_CoM.x() + lin_vel_d_world.x() * t,
            input.p_CoM.y() + lin_vel_d_world.y() * t,
            input.com_height_d,
            ang_vel_d_world,
            lin_vel_d_world,
            -9.8;

        calculateAMatC(mpc_states_d.segment<3>(i * kStateDim));
        calculateBMatC(input.mass, input.trunk_inertia, input.R_B, input.contact_pos_abs);
        stateSpaceDiscretization(dt);
        B_mat_d_list_.block<kStateDim, kForceDim>(i * kStateDim, 0) = B_mat_d_;
    }

    calculateQPMats(input);

    Eigen::MatrixXd weighted_b_qp = B_qp_;
    weighted_b_qp.array().colwise() *= Q_.diagonal().array();

    Eigen::MatrixXd dense_hessian(kNumDecisionVars, kNumDecisionVars);
    dense_hessian.noalias() = B_qp_.transpose() * weighted_b_qp;
    dense_hessian.diagonal() += R_.diagonal();
    dense_hessian = 0.5 * (dense_hessian + dense_hessian.transpose());
    dense_hessian.diagonal().array() += 1e-9;
    const Eigen::SparseMatrix<double> hessian = sparseFromDense(dense_hessian);

    const Eigen::Matrix<double, kStateDim * kPlanHorizon, 1> state_error =
        A_qp_ * mpc_states - mpc_states_d;
    const Eigen::Matrix<double, kNumDecisionVars, 1> gradient =
        B_qp_.transpose() * (Q_.diagonal().array() * state_error.array()).matrix();

    Eigen::Matrix<double, kNumDecisionVars, 1> solution;
    if (!solveQP(hessian, gradient, solution)) {
        return false;
    }

    grf_ = solution.segment<kForceDim>(0);
    predicted_states_ = A_qp_ * mpc_states + B_qp_ * solution;

    return grf_.allFinite();
}

void ConvexMpc::calculateAMatC(const Eigen::Vector3d& root_euler)
{
    A_mat_c_.setZero();

    const double cos_yaw = std::cos(root_euler.z());
    const double sin_yaw = std::sin(root_euler.z());

    Eigen::Matrix3d ang_vel_to_rpy_rate;
    ang_vel_to_rpy_rate << cos_yaw, sin_yaw, 0.0,
                          -sin_yaw, cos_yaw, 0.0,
                          0.0, 0.0, 1.0;

    A_mat_c_.block<3, 3>(0, 6) = ang_vel_to_rpy_rate;
    A_mat_c_.block<3, 3>(3, 9).setIdentity();
    A_mat_c_(11, 12) = 1.0;
}

void ConvexMpc::calculateBMatC(
    double robot_mass,
    const Eigen::Matrix3d& trunk_inertia,
    const Eigen::Matrix3d& root_rot_mat,
    const Eigen::Matrix<double, 3, kNumContacts>& foot_pos)
{
    B_mat_c_.setZero();

    const Eigen::Matrix3d trunk_inertia_world =
        root_rot_mat * trunk_inertia * root_rot_mat.transpose();
    const Eigen::Matrix3d trunk_inertia_inv =
        trunk_inertia_world.completeOrthogonalDecomposition().pseudoInverse();

    for (int i = 0; i < kNumContacts; ++i) {
        B_mat_c_.block<3, 3>(6, 3 * i) =
            trunk_inertia_inv * skew(foot_pos.block<3, 1>(0, i));
        B_mat_c_.block<3, 3>(9, 3 * i) =
            (1.0 / robot_mass) * Eigen::Matrix3d::Identity();
    }
}

void ConvexMpc::stateSpaceDiscretization(double dt)
{
    A_mat_d_ = Eigen::Matrix<double, kStateDim, kStateDim>::Identity() + A_mat_c_ * dt;
    B_mat_d_ = B_mat_c_ * dt;
}

void ConvexMpc::calculateQPMats(const Input& input)
{
    A_qp_.setZero();
    B_qp_.setZero();

    for (int i = 0; i < kPlanHorizon; ++i) {
        if (i == 0) {
            A_qp_.block<kStateDim, kStateDim>(i * kStateDim, 0) = A_mat_d_;
        } else {
            A_qp_.block<kStateDim, kStateDim>(i * kStateDim, 0) =
                A_qp_.block<kStateDim, kStateDim>((i - 1) * kStateDim, 0) * A_mat_d_;
        }

        for (int j = 0; j < i + 1; ++j) {
            if (i - j == 0) {
                B_qp_.block<kStateDim, kForceDim>(i * kStateDim, j * kForceDim) =
                    B_mat_d_list_.block<kStateDim, kForceDim>(j * kStateDim, 0);
            } else {
                B_qp_.block<kStateDim, kForceDim>(i * kStateDim, j * kForceDim) =
                    A_qp_.block<kStateDim, kStateDim>((i - j - 1) * kStateDim, 0)
                    * B_mat_d_list_.block<kStateDim, kForceDim>(j * kStateDim, 0);
            }
        }
    }

    for (int i = 0; i < kPlanHorizon; ++i) {
        for (int j = 0; j < kNumContacts; ++j) {
            const double contact = static_cast<double>(input.contact_schedule(j, i));
            const int offset = i * kConstraintDim + j * 5;
            lb_.segment<5>(offset) << 0.0,
                                      -OsqpEigen::INFTY,
                                      0.0,
                                      -OsqpEigen::INFTY,
                                      fz_min_ * contact;
            ub_.segment<5>(offset) << OsqpEigen::INFTY,
                                      0.0,
                                      OsqpEigen::INFTY,
                                      0.0,
                                      fz_max_ * contact;
        }
    }
}

bool ConvexMpc::solveQP(
    const Eigen::SparseMatrix<double>& hessian,
    const Eigen::Matrix<double, kNumDecisionVars, 1>& gradient,
    Eigen::Matrix<double, kNumDecisionVars, 1>& solution)
{
    gradient_osqp_ = gradient;
    lb_osqp_ = lb_;
    ub_osqp_ = ub_;

    if (!solver_initialized_) {
        solver_.settings()->setVerbosity(false);
        solver_.settings()->setWarmStart(true);
        solver_.settings()->setMaxIteration(1000);
        solver_.settings()->setAbsoluteTolerance(1e-4);
        solver_.settings()->setRelativeTolerance(1e-4);

        solver_.data()->setNumberOfVariables(kNumDecisionVars);
        solver_.data()->setNumberOfConstraints(kNumConstraints);
        if (!solver_.data()->setHessianMatrix(hessian)
            || !solver_.data()->setGradient(gradient_osqp_)
            || !solver_.data()->setLinearConstraintsMatrix(constraint_mat_)
            || !solver_.data()->setLowerBound(lb_osqp_)
            || !solver_.data()->setUpperBound(ub_osqp_)) {
            std::cerr << "ConvexMPC OSQP data setup failed" << std::endl;
            return false;
        }

        if (!solver_.initSolver()) {
            std::cerr << "ConvexMPC OSQP init failed" << std::endl;
            return false;
        }
        solver_initialized_ = true;
    } else {
        const bool updated =
            solver_.updateHessianMatrix(hessian)
            && solver_.updateGradient(gradient_osqp_)
            && solver_.updateBounds(lb_osqp_, ub_osqp_);
        if (!updated) {
            std::cerr << "ConvexMPC OSQP update failed" << std::endl;
            solver_.clearSolver();
            solver_initialized_ = false;
            return solveQP(hessian, gradient, solution);
        }
    }

    const auto exit_flag = solver_.solveProblem();
    const auto status = solver_.getStatus();
    if (exit_flag != OsqpEigen::ErrorExitFlag::NoError
        || (status != OsqpEigen::Status::Solved
            && status != OsqpEigen::Status::SolvedInaccurate)) {
        std::cerr << "ConvexMPC OSQP solve failed: exit="
                  << static_cast<int>(exit_flag)
                  << ", status=" << static_cast<int>(status) << std::endl;
        return false;
    }

    solution = solver_.getSolution();
    return solution.allFinite();
}

#if 0
void ConvexMpc::reset() 
{
    // continuous time state space model
    A_mat_c.setZero();
    B_mat_c.setZero();
    B_mat_c_list.setZero();
    AB_mat_c.setZero();
    
    // discrete time state space model
    A_mat_d.setZero();
    B_mat_d.setZero();
    B_mat_d_list.setZero();
    
    // MPC state space model
    AB_mat_d.setZero();
    A_qp.setZero();
    B_qp.setZero();
    gradient.setZero();
    lb.setZero();
    ub.setZero();
}
void ConvexMpc::calculate_A_mat_c(Eigen::Vector3d root_euler) 
{
    // std::cout << "yaw: " << root_euler[2] << std::endl;
    double cos_yaw = cos(root_euler[2]);
    double sin_yaw = sin(root_euler[2]);

    Eigen::Matrix3d ang_vel_to_rpy_rate;

    ang_vel_to_rpy_rate << cos_yaw, sin_yaw, 0,
                            -sin_yaw, cos_yaw, 0,
                            0, 0, 1;

    A_mat_c.block<3, 3>(0, 6) = ang_vel_to_rpy_rate;
    A_mat_c.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity();
    A_mat_c(11, NUM_DOF) = 1;
}

void ConvexMpc::calculate_B_mat_c(double robot_mass, const Eigen::Matrix3d &trunk_inertia, Eigen::Matrix3d root_rot_mat,
                                  Eigen::Matrix<double, 3, NUM_LEG> foot_pos) 
{
    // we need to calculate PLAN_HORIZON B matrices
    Eigen::Matrix3d trunk_inertia_world;
    trunk_inertia_world = root_rot_mat * trunk_inertia * root_rot_mat.transpose();
    for (int i = 0; i < NUM_LEG; ++i) 
    {
        B_mat_c.block<3, 3>(6, 3 * i) = trunk_inertia_world.inverse() * Utils::skew(foot_pos.block<3, 1>(0, i));    // right toe -> right wheel -> left toe -> left wheel
        B_mat_c.block<3, 3>(9, 3 * i) = (1 / robot_mass) * Eigen::Matrix3d::Identity();
    }
}
#endif

#if 0
void ConvexMpc::state_space_discretization(double dt) 
{
    // simplified exp 
    // AB_mat_d = (dt * AB_mat_c).exp();
    A_mat_d = Eigen::Matrix<double, MPC_STATE_DIM, MPC_STATE_DIM>::Identity() + A_mat_c * dt;
    B_mat_d = B_mat_c * dt;
}

void ConvexMpc::calculate_qp_mats(RobotState &state, Eigen::Vector<double, MPC_STATE_DIM> mpc_states, \
                            Eigen::Vector<double, MPC_STATE_DIM * PLAN_HORIZON> mpc_states_d)
{

    // reserve size for sparse matrix
    Q_sparse = Eigen::SparseMatrix<double>(MPC_STATE_DIM * PLAN_HORIZON,MPC_STATE_DIM * PLAN_HORIZON);
    R_sparse = Eigen::SparseMatrix<double>(NUM_DOF * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON);

    q_weights_mpc.resize(MPC_STATE_DIM * PLAN_HORIZON);
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        q_weights_mpc.segment(i * MPC_STATE_DIM, MPC_STATE_DIM) = state.param.q_weights;
    }
    Q.diagonal() = 2*q_weights_mpc;
    for (int i = 0; i < MPC_STATE_DIM*PLAN_HORIZON; ++i) {
        Q_sparse.insert(i,i) = 2*q_weights_mpc(i);
    }

    r_weights_mpc.resize(NUM_DOF * PLAN_HORIZON);
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        r_weights_mpc.segment(i * NUM_DOF, NUM_DOF) = state.param.r_weights;
    }
    R.diagonal() = 2*r_weights_mpc;
    for (int i = 0; i < NUM_DOF*PLAN_HORIZON; ++i) {
        R_sparse.insert(i,i) = 2*r_weights_mpc(i);
    }

    // calculate A_qp and B_qp
    Eigen::Matrix<double, MPC_STATE_DIM, NUM_DOF> tmp_mtx;

    // keep A_qp as a storage list 
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        if (i == 0) {
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * i, 0) = A_mat_d;
        }
        else {
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * i, 0) = 
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * (i-1), 0) * A_mat_d;
        }
        for (int j = 0; j < i + 1; ++j) {
            if (i-j == 0) {
                B_qp.block<MPC_STATE_DIM, NUM_DOF>(MPC_STATE_DIM * i, NUM_DOF * j) =
                    B_mat_d_list.block<MPC_STATE_DIM, NUM_DOF>(j * MPC_STATE_DIM, 0);
            } else {
                B_qp.block<MPC_STATE_DIM, NUM_DOF>(MPC_STATE_DIM * i, NUM_DOF * j) =
                        A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * (i-j-1), 0) 
                        * B_mat_d_list.block<MPC_STATE_DIM, NUM_DOF>(j * MPC_STATE_DIM, 0);
            }
        }
    }

    // calculate hessian
    Eigen::Matrix<double, NUM_DOF * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON> dense_hessian;
    // dense_hessian = (B_qp.transpose() * Q * B_qp + R);
    dense_hessian = (B_qp.transpose() * Q * B_qp);
    dense_hessian += R;
    hessian = dense_hessian.sparseView();


    // calculate gradient
    Eigen::Matrix<double, 13 * PLAN_HORIZON, 1> tmp_vec = A_qp * mpc_states;
    tmp_vec -= mpc_states_d;
    gradient = B_qp.transpose() * Q * tmp_vec;

    // calculate lower bound and upper bound
    fz_min = 0;
    fz_max = 500;

    // calculate linear constraints
    Eigen::VectorXd lb_one_horizon(MPC_CONSTRAINT_DIM);
    Eigen::VectorXd ub_one_horizon(MPC_CONSTRAINT_DIM);

    for(int i=0; i<PLAN_HORIZON; i++)
    {
        for (int j = 0; j < NUM_LEG; j++) 
        {
            lb_one_horizon.segment<5>(j * 5) << 0,
                    -OsqpEigen::INFTY,
                    0,
                    -OsqpEigen::INFTY,
                    fz_min *  state.ctrl.contact_window[j][i];
            ub_one_horizon.segment<5>(j * 5) << OsqpEigen::INFTY,
                    0,
                    OsqpEigen::INFTY,
                    0,
                    fz_max *  state.ctrl.contact_window[j][i];
        }
        lb.segment<MPC_CONSTRAINT_DIM>(i * MPC_CONSTRAINT_DIM) = lb_one_horizon;
        ub.segment<MPC_CONSTRAINT_DIM>(i * MPC_CONSTRAINT_DIM) = ub_one_horizon;
    }
}
#endif
#if 0
void ConvexMpc::calculate_qp_mats(A1CtrlStates state)
{
    // standard QP formulation
    // minimize 1/2 * x' * P * x + q' * x
    // subject to lb <= Ac * x <= ub
    //
    // A_qp = [A,
    //         A^2,
    //         A^3,
    //         ...
    //         A^k]'
    //
    // B_qp = [A^0*B(0),
    //         A^1*B(0),     B(1),
    //         A^2*B(0),     A*B(1),       B(2),
    //         ...
    //         A^(k-1)*B(0), A^(k-2)*B(1), A^(k-3)*B(2), ... B(k-1)]

    // calculate A_qp and B_qp
    Eigen::Matrix<double, MPC_STATE_DIM, NUM_DOF> tmp_mtx;

    // keep A_qp as a storage list 
    for (int i = 0; i < PLAN_HORIZON; ++i) {
        if (i == 0) {
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * i, 0) = A_mat_d;
        }
        else {
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * i, 0) = 
            A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * (i-1), 0) * A_mat_d;
        }
        for (int j = 0; j < i + 1; ++j) {
            if (i-j == 0) {
                B_qp.block<MPC_STATE_DIM, NUM_DOF>(MPC_STATE_DIM * i, NUM_DOF * j) =
                    B_mat_d_list.block<MPC_STATE_DIM, NUM_DOF>(j * MPC_STATE_DIM, 0);
            } else {
                B_qp.block<MPC_STATE_DIM, NUM_DOF>(MPC_STATE_DIM * i, NUM_DOF * j) =
                        A_qp.block<MPC_STATE_DIM, MPC_STATE_DIM>(MPC_STATE_DIM * (i-j-1), 0) 
                        * B_mat_d_list.block<MPC_STATE_DIM, NUM_DOF>(j * MPC_STATE_DIM, 0);
            }
        }
    }

    // calculate hessian
    Eigen::Matrix<double, NUM_DOF * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON> dense_hessian;
    // dense_hessian = (B_qp.transpose() * Q * B_qp + R);
    dense_hessian = (B_qp.transpose() * Q * B_qp);
    dense_hessian += R;
    hessian = dense_hessian.sparseView();

    // calculate gradient
    Eigen::Matrix<double, 13 * PLAN_HORIZON, 1> tmp_vec = A_qp* state.mpc_states;
    tmp_vec -= state.mpc_states_d;
    gradient = B_qp.transpose() * Q * tmp_vec;

    // calculate lower bound and upper bound
    fz_min = 0;
    fz_max = 800;

    // calculate linear constraints
    Eigen::VectorXd lb_one_horizon(MPC_CONSTRAINT_DIM);
    Eigen::VectorXd ub_one_horizon(MPC_CONSTRAINT_DIM);
    for(int i=0; i<PLAN_HORIZON; i++)
    {
        for (int j = 0; j < NUM_LEG; j++) 
        {
            // std::cout << state.contact_window[j][i] << std::endl;
            lb_one_horizon.segment<5>(j * 5) << 0,
                    -OsqpEigen::INFTY,
                    0,
                    -OsqpEigen::INFTY,
                    fz_min *  state.contact_window[j][i];
            ub_one_horizon.segment<5>(j * 5) << OsqpEigen::INFTY,
                    0,
                    OsqpEigen::INFTY,
                    0,
                    fz_max *  state.contact_window[j][i];
            }
        lb.segment<MPC_CONSTRAINT_DIM>(i * MPC_CONSTRAINT_DIM) = lb_one_horizon;
        ub.segment<MPC_CONSTRAINT_DIM>(i * MPC_CONSTRAINT_DIM) = ub_one_horizon;
    }
}
#endif
