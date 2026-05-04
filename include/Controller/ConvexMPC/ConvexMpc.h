#if 0
//
// Created by zixin on 12/09/21.
//

#ifndef A1_CPP_CONVEXMPC_H
#define A1_CPP_CONVEXMPC_H

// #define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <vector>
#include <chrono>

// #define EIGEN_DONT_ALIGN
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#include "MahruCtrlStates.h"
#include "Robot_States.h"
#include "Robot_Params.h"
#include "utils/Utils.h"

class ConvexMpc {
public:
    ConvexMpc(Eigen::VectorXd &q_weights_, Eigen::VectorXd &r_weights_);

    void reset();

    void calculate_A_mat_c(Eigen::Vector3d root_euler);

    void calculate_B_mat_c(double robot_mass, const Eigen::Matrix3d &a1_trunk_inertia, Eigen::Matrix3d root_rot_mat,
                           Eigen::Matrix<double, 3, NUM_LEG> foot_pos);

    void state_space_discretization(double dt);

    void calculate_qp_mats(A1CtrlStates state);
    void calculate_qp_mats(RobotState &state, Eigen::Vector<double, MPC_STATE_DIM> mpc_states, \
                                Eigen::Vector<double, MPC_STATE_DIM * PLAN_HORIZON> mpc_states_d);
    // void calculate_qp_mats(Eigen::Matrix<double,MPC_STATE_DIM,1> mpc_states, Eigen::Matrix<double,PLAN_HORIZON*MPC_STATE_DIM,1> mpc_states_d, bool contacts[NUM_LEG]);

//private:
    // parameters initialized with class ConvexMpc
    double mu;
    double fz_min;
    double fz_max;

    // Eigen::VectorXd q_weights_mpc; // (state_dim * horizon) x 1
    // Eigen::VectorXd r_weights_mpc; // (action_dim * horizon) x 1
    Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, 1> q_weights_mpc;
    Eigen::Matrix<double, NUM_DOF * PLAN_HORIZON, 1> r_weights_mpc;

    Eigen::DiagonalMatrix<double, MPC_STATE_DIM * PLAN_HORIZON> Q;
    Eigen::DiagonalMatrix<double, NUM_DOF * PLAN_HORIZON> R;
    Eigen::SparseMatrix<double> Q_sparse;
    Eigen::SparseMatrix<double> R_sparse;
    // Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, MPC_STATE_DIM * PLAN_HORIZON> Q;
    // Eigen::Matrix<double, NUM_DOF * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON> R;

    // parameters initialized in the function reset()
    // Eigen::MatrixXd A_mat_c;
    // Eigen::MatrixXd B_mat_c;
    // Eigen::MatrixXd AB_mat_c;

    // Eigen::MatrixXd A_mat_d;
    // Eigen::MatrixXd B_mat_d;
    // Eigen::MatrixXd AB_mat_d;

    // Eigen::MatrixXd A_qp;
    // Eigen::MatrixXd B_qp;

    Eigen::Matrix<double, MPC_STATE_DIM, MPC_STATE_DIM> A_mat_c;
    Eigen::Matrix<double, MPC_STATE_DIM, NUM_DOF> B_mat_c;
    Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, NUM_DOF> B_mat_c_list;
    Eigen::Matrix<double, MPC_STATE_DIM + NUM_DOF, MPC_STATE_DIM + NUM_DOF> AB_mat_c;

    Eigen::Matrix<double, MPC_STATE_DIM, MPC_STATE_DIM> A_mat_d;
    Eigen::Matrix<double, MPC_STATE_DIM, NUM_DOF> B_mat_d;
    Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, NUM_DOF> B_mat_d_list;
    Eigen::Matrix<double, MPC_STATE_DIM + NUM_DOF, MPC_STATE_DIM + NUM_DOF> AB_mat_d;

    Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, MPC_STATE_DIM> A_qp;
    Eigen::Matrix<double, MPC_STATE_DIM * PLAN_HORIZON, NUM_DOF * PLAN_HORIZON> B_qp;

    // standard QP formulation
    // minimize 1/2 * x' * P * x + q' * x
    // subject to lb <= Ac * x <= ub
    Eigen::SparseMatrix<double> hessian; // P
    Eigen::SparseMatrix<double> linear_constraints; // Ac
//    Eigen::VectorXd gradient; // q
//    Eigen::VectorXd lb;
//    Eigen::VectorXd ub;
    Eigen::Matrix<double, NUM_DOF * PLAN_HORIZON, 1> gradient; // q
    Eigen::Matrix<double, MPC_CONSTRAINT_DIM * PLAN_HORIZON, 1> lb;
    Eigen::Matrix<double, MPC_CONSTRAINT_DIM * PLAN_HORIZON, 1> ub;

};

#endif //A1_CPP_CONVEXMPC_H
#endif

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

class ConvexMpc {
public:
    static constexpr int kPlanHorizon = 10;
    static constexpr int kStateDim = 13;
    static constexpr int kNumContacts = 4;
    static constexpr int kForceDim = 3 * kNumContacts;
    static constexpr int kConstraintDim = 5 * kNumContacts;
    static constexpr int kNumDecisionVars = kForceDim * kPlanHorizon;
    static constexpr int kNumConstraints = kConstraintDim * kPlanHorizon;

    struct Input {
        double mass = 1.0;
        Eigen::Matrix3d trunk_inertia = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_B = Eigen::Matrix3d::Identity();
        Eigen::Vector3d euler_B = Eigen::Vector3d::Zero();
        Eigen::Vector3d p_CoM = Eigen::Vector3d::Zero();
        Eigen::Vector3d pdot_CoM = Eigen::Vector3d::Zero();
        Eigen::Vector3d omega_B = Eigen::Vector3d::Zero();
        Eigen::Vector3d lin_vel_d = Eigen::Vector3d::Zero();
        Eigen::Vector3d ang_vel_d = Eigen::Vector3d::Zero();
        Eigen::Vector3d euler_B_d = Eigen::Vector3d::Zero();
        double com_height_d = 0.0;
        Eigen::Matrix<double, 3, kNumContacts> contact_pos_abs =
            Eigen::Matrix<double, 3, kNumContacts>::Zero();
        Eigen::Matrix<double, 3, kNumContacts * kPlanHorizon> contact_pos_world_horizon =
            Eigen::Matrix<double, 3, kNumContacts * kPlanHorizon>::Zero();
        bool use_contact_pos_world_horizon = false;
        Eigen::Matrix<double, kStateDim * kPlanHorizon, 1> state_ref_horizon =
            Eigen::Matrix<double, kStateDim * kPlanHorizon, 1>::Zero();
        bool use_state_ref_horizon = false;
        Eigen::Matrix<int, kNumContacts, kPlanHorizon> contact_schedule =
            Eigen::Matrix<int, kNumContacts, kPlanHorizon>::Ones();
    };

    ConvexMpc();

    bool update(const Input& input, double dt);
    void reset();

    const Eigen::Matrix<double, kForceDim, 1>& groundReactionForce() const
    {
        return grf_;
    }

    const Eigen::Matrix<double, kStateDim * kPlanHorizon, 1>& predictedStates() const
    {
        return predicted_states_;
    }

private:
    double mu_ = 0.7;
    double fz_min_ = 0.0;
    double fz_max_ = 500.0;

    Eigen::Matrix<double, kStateDim, 1> q_weights_;
    Eigen::Matrix<double, kForceDim, 1> r_weights_;
    Eigen::DiagonalMatrix<double, kStateDim * kPlanHorizon> Q_;
    Eigen::DiagonalMatrix<double, kNumDecisionVars> R_;

    Eigen::Matrix<double, kStateDim, kStateDim> A_mat_c_;
    Eigen::Matrix<double, kStateDim, kForceDim> B_mat_c_;
    Eigen::Matrix<double, kStateDim * kPlanHorizon, kStateDim> A_mat_d_list_;
    Eigen::Matrix<double, kStateDim * kPlanHorizon, kForceDim> B_mat_d_list_;
    Eigen::Matrix<double, kStateDim, kStateDim> A_mat_d_;
    Eigen::Matrix<double, kStateDim, kForceDim> B_mat_d_;
    Eigen::Matrix<double, kStateDim * kPlanHorizon, kStateDim> A_qp_;
    Eigen::Matrix<double, kStateDim * kPlanHorizon, kNumDecisionVars> B_qp_;

    Eigen::SparseMatrix<double> constraint_mat_;
    Eigen::Matrix<double, kNumConstraints, 1> lb_;
    Eigen::Matrix<double, kNumConstraints, 1> ub_;
    Eigen::Matrix<double, kForceDim, 1> grf_ =
        Eigen::Matrix<double, kForceDim, 1>::Zero();
    Eigen::Matrix<double, kStateDim * kPlanHorizon, 1> predicted_states_ =
        Eigen::Matrix<double, kStateDim * kPlanHorizon, 1>::Zero();

    OsqpEigen::Solver solver_;
    bool solver_initialized_ = false;
    Eigen::VectorXd gradient_osqp_;
    Eigen::VectorXd lb_osqp_;
    Eigen::VectorXd ub_osqp_;

    void calculateAMatC(const Eigen::Vector3d& root_euler);
    void calculateBMatC(double robot_mass,
                        const Eigen::Matrix3d& trunk_inertia,
                        const Eigen::Matrix3d& root_rot_mat,
                        const Eigen::Matrix<double, 3, kNumContacts>& foot_pos);
    void stateSpaceDiscretization(double dt);
    void calculateQPMats(const Input& input);
    bool solveQP(const Eigen::SparseMatrix<double>& hessian,
                 const Eigen::Matrix<double, kNumDecisionVars, 1>& gradient,
                 Eigen::Matrix<double, kNumDecisionVars, 1>& solution);
};
