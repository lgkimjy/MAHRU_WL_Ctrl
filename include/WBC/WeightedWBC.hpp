#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "3rd-parties/ARBMLlib/ARBML.h"
#include "Controller/ConvexMPC/ConvexMpc.h"
#include "RobotDefinition.hpp"
#include "RobotStates.hpp"
#include "WBC/WBCTask.hpp"

class WeightedWBC
{
public:
    struct Input {
        Eigen::Matrix<double, mahru::num_act_joint, 1> q_d =
            Eigen::Matrix<double, mahru::num_act_joint, 1>::Zero();
        Eigen::Matrix<double, mahru::num_act_joint, 1> qdot_d =
            Eigen::Matrix<double, mahru::num_act_joint, 1>::Zero();
        Eigen::Matrix<double, mahru::num_act_joint, 1> qddot_d =
            Eigen::Matrix<double, mahru::num_act_joint, 1>::Zero();
        Eigen::Vector3d p_CoM_d = Eigen::Vector3d::Zero();
        Eigen::Vector3d pdot_CoM_d = Eigen::Vector3d::Zero();
        bool enable_horizontal_com_task = false;
        Eigen::Vector3d lin_vel_d = Eigen::Vector3d::Zero();
        Eigen::Vector3d ang_vel_d = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_B_d = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1> grfs_mpc =
            Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();
        int swing_contact_index = -1;
        int previous_state_machine = 0;
        Eigen::Vector3d p_sw_d = Eigen::Vector3d::Zero();
        Eigen::Vector3d pdot_sw_d = Eigen::Vector3d::Zero();
        bool enable_sliding_task = false;
        Eigen::Matrix<double, 3, ConvexMpc::kNumContacts> p_slide_d =
            Eigen::Matrix<double, 3, ConvexMpc::kNumContacts>::Zero();
        Eigen::Matrix<double, 3, ConvexMpc::kNumContacts> pdot_slide_d =
            Eigen::Matrix<double, 3, ConvexMpc::kNumContacts>::Zero();
        Eigen::Vector3d sliding_axis_mask = Eigen::Vector3d(1.0, 1.0, 0.0);
        double sliding_kp = 400.0;
        double sliding_kd = 40.0;
        bool enable_upper_body_momentum_compensation_task = false;
        Eigen::Vector3d upper_body_momentum_axis_mask =
            Eigen::Vector3d(0.0, 0.0, 1.0);
        double upper_body_momentum_kp = 8.0;
        double upper_body_momentum_max_rate = 25.0;
        double upper_body_momentum_task_scale = 1.0;
        double upper_body_posture_kp = 30.0;
        double upper_body_posture_kd = 6.0;
    };

    struct Output {
        bool solved = false;
        Eigen::Matrix<double, mahru::nDoF, 1> xiddot_d =
            Eigen::Matrix<double, mahru::nDoF, 1>::Zero();
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1> grfs_d =
            Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();
        Eigen::Matrix<double, mahru::num_act_joint, 1> torq_ff =
            Eigen::Matrix<double, mahru::num_act_joint, 1>::Zero();
    };

    struct Diagnostics {
        bool solved = false;
        int num_contact_points = 0;
        double contact_jacobian_norm = 0.0;
        double contact_jacobian_max = 0.0;
        double swing_jacobian_norm = 0.0;
        double xiddot_max = 0.0;
        double base_xiddot_max = 0.0;
        double joint_qddot_max = 0.0;
        int joint_qddot_max_index = -1;
        double delta_grf_max = 0.0;
        double torque_norm = 0.0;
        double torque_max = 0.0;
        double joint_qddot_limit = 0.0;
        int sliding_wheel_joint = -1;
        double sliding_vel_error = 0.0;
        double sliding_qddot_cmd = 0.0;
        double sliding_contact_error_norm = 0.0;
        double rolling_contact_vel_error = 0.0;
        double rolling_contact_acc_cmd = 0.0;
        int swing_wheel_joint = -1;
        int contact_normal_rows = 0;
        double swing_leg_angular_momentum_norm = 0.0;
        double upper_body_angular_momentum_norm = 0.0;
        double upper_body_momentum_task_max = 0.0;
        double pelvis_yaw_error = 0.0;
        double pelvis_yaw_acc_cmd = 0.0;
    };

    WeightedWBC();
    ~WeightedWBC() { std::cout << "WeightedWBC Destructor" << std::endl; }

    Output update(CARBML& robot, const RobotData& state, const Input& input);
    void loadWeightGain();
    const Diagnostics& diagnostics() const { return diagnostics_; }

    int numContactPoints_ = 4;

private:
    static constexpr int kNumDecisionVars = mahru::nDoF + ConvexMpc::kForceDim;

    const CARBML* robot_ = nullptr;
    const RobotData* state_ = nullptr;
    Input input_;

    Eigen::MatrixXd Sf_;
    Eigen::Matrix<double, ConvexMpc::kForceDim, mahru::nDoF> Jp_contact_ =
        Eigen::Matrix<double, ConvexMpc::kForceDim, mahru::nDoF>::Zero();
    Eigen::Matrix<double, ConvexMpc::kForceDim, mahru::nDoF> Jdotp_contact_ =
        Eigen::Matrix<double, ConvexMpc::kForceDim, mahru::nDoF>::Zero();
    Eigen::Matrix<double, DOF3, mahru::nDoF> Jp_sw_ =
        Eigen::Matrix<double, DOF3, mahru::nDoF>::Zero();
    Eigen::Matrix<double, DOF3, mahru::nDoF> Jdotp_sw_ =
        Eigen::Matrix<double, DOF3, mahru::nDoF>::Zero();
    Eigen::Vector3d p_sw_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d pdot_sw_ = Eigen::Vector3d::Zero();

    double W_swingLeg_ = 10.0;
    double W_JointAcc_ = 1.0;
    double W_qddot_regul_ = 1e-3;
    double W_rf_regul_ = 1e-3;
    double W_centroidal_ = 100.0;
    double W_CenAngMom_Compen_ = 1.0;
    double W_wheelAccel_ = 10.0;
    double W_sliding_contact_ = 10.0;
    double W_rolling_contact_ = 20.0;
    double W_pelvis_yaw_ = 0.0;
    double W_upper_body_momentum_compensation_ = 0.0;
    double W_upper_body_posture_ = 0.0;
    double joint_qddot_limit_ = 120.0;
    Eigen::Vector3d sliding_contact_axis_mask_ =
        Eigen::Vector3d(1.0, 1.0, 0.0);
    double sliding_contact_kp_ = 120.0;
    double sliding_contact_kd_ = 25.0;
    double sliding_contact_acc_limit_ = 80.0;
    double rolling_contact_kd_ = 40.0;
    double rolling_contact_acc_limit_ = 120.0;
    double pelvis_yaw_kp_ = 120.0;
    double pelvis_yaw_kd_ = 20.0;
    double pelvis_yaw_acc_limit_ = 80.0;

    Eigen::Matrix3d kp_CoM_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d kd_CoM_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d kp_R_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d kd_omega_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d kp_swing_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d kd_swing_ = Eigen::Matrix3d::Zero();

    std::vector<int> selectedJointsIdx_;
    std::vector<int> upperBodyMomentumJoints_ = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    Eigen::VectorXd kp_jacc_;
    Eigen::VectorXd kd_jacc_;

    double K_f_ = 1.0;
    double K_mu_ = 0.8;
    Diagnostics diagnostics_;

    void reconfigStates();
    void configureContacts();
    WBCTask formulateConstraints();
    WBCTask formulateWeightedTask();

    WBCTask formulateLinearMotionTask();
    WBCTask formulatePelvisYawTask();
    WBCTask formulateUpperBodyMomentumCompensationTask();
    WBCTask formulateUpperBodyPostureTask();
    WBCTask formulateSwingLegTask(const Eigen::Matrix3d& swingKp,
                                  const Eigen::Matrix3d& swingKd);
    WBCTask formulateSlidingJointTask();
    WBCTask formulateSlidingContactTask();
    WBCTask formulateSupportWheelTask();
    WBCTask formulateSwingWheelTask();
    WBCTask formulateRollingContactTask();
    WBCTask formulateJointAccelerationTask(const std::vector<int>& selectedJointsIdx,
                                           const Eigen::VectorXd& Kp,
                                           const Eigen::VectorXd& Kd);
    WBCTask formulateRegularizationTask(double qddot_regul, double rf_regul);

    WBCTask formulateFloatingBaseConstraint();
    WBCTask formulateContactNormalConstraint();
    WBCTask formulateFrictionConeConstraint();
    WBCTask formulateAccelerationLimitConstraint();
    bool solveQP(const WBCTask& constraints,
                 const WBCTask& weightedTask,
                 Eigen::VectorXd& qpSol) const;
};
