#ifndef __FSM_UNICYCLECTRL_HPP__
#define __FSM_UNICYCLECTRL_HPP__

#include <string>
#include <mujoco/mujoco.h>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

#include "3rd-parties/ARBMLlib/ARBML.h"
#include "Controller/ConvexMPC/ConvexMpc.h"
#include "WBC/WeightedWBC.hpp"

using namespace mahru;

template <typename T>
class FSM_UnicycleCtrlState : public States {
public:
    explicit FSM_UnicycleCtrlState(RobotData& robot);
    ~FSM_UnicycleCtrlState() { delete arbml_; }

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {};
    void runTransition() override {};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData*  robot_data_;
    CARBML*     arbml_ = nullptr;
    mujoco::TrajVizUtil* viz_ = nullptr;
    WeightedWBC weighted_wbc_;

    Eigen::Matrix<T, num_act_joint, 1>      jpos_0_;
    Eigen::Vector3d                         p_CoM_nominal_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         p_CoM_wbc_d_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         pdot_CoM_wbc_d_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         pddot_CoM_wbc_ff_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d                         R_B_wbc_d_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix<T, num_act_joint, 1>      torq_wbc_ =
        Eigen::Matrix<T, num_act_joint, 1>::Zero();
    Eigen::Matrix<double, ConvexMpc::kForceDim, 1> nominal_grf_ =
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();
    bool                                    is_wbc_solved_ = false;
    bool                                    enable_sway_ = true;
    Eigen::Vector3d                         sway_amplitude_ = Eigen::Vector3d::Zero();
    double                                  sway_frequency_hz_ = 0.1;
    double                                  sway_ramp_time_ = 1.0;

    bool                                    enable_com_shift_ = true;
    double                                  com_shift_duration_ = 3.0;
    Eigen::Vector3d                         com_shift_offset_ = Eigen::Vector3d::Zero();
    bool                                    com_shift_initialized_ = false;
    Eigen::Vector3d                         com_shift_start_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         com_shift_target_ = Eigen::Vector3d::Zero();

    bool                                    enable_right_foot_lift_ = true;
    double                                  right_foot_lift_start_time_ = 3.0;
    double                                  right_foot_lift_height_ = 0.04;
    double                                  right_foot_lift_duration_ = 1.0;
    bool                                    right_foot_lift_initialized_ = false;
    Eigen::Vector3d                         right_wheel_lift_start_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_toe_lift_start_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_wheel_lift_pelvis_offset_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_toe_lift_pelvis_offset_ = Eigen::Vector3d::Zero();
    bool                                    right_wheel_lift_pelvis_offset_configured_ = false;
    Eigen::Vector3d                         right_wheel_lift_ref_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_wheel_lift_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_toe_lift_ref_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         right_toe_lift_vel_ = Eigen::Vector3d::Zero();
    double                                  swing_clearance_height_ = 0.04;
    double                                  swing_clearance_kp_ = 600.0;
    double                                  swing_clearance_kd_ = 50.0;
    double                                  swing_clearance_max_acc_ = 120.0;
    double                                  swing_lateral_clearance_distance_ = 0.25;
    double                                  swing_lateral_clearance_kp_ = 200.0;
    double                                  swing_lateral_clearance_kd_ = 20.0;
    double                                  swing_lateral_clearance_max_acc_ = 50.0;
    bool                                    enable_swing_leg_reaction_ = true;
    double                                  swing_leg_reaction_sign_ = -1.0;
    double                                  swing_leg_reaction_kp_ = 0.45;
    double                                  swing_leg_reaction_kd_ = 0.08;
    double                                  swing_leg_reaction_max_offset_ = 0.16;
    double                                  swing_leg_reaction_max_vel_ = 1.2;
    double                                  swing_leg_reaction_tau_ = 0.07;
    double                                  swing_leg_reaction_offset_ = 0.0;
    double                                  swing_leg_reaction_vel_ = 0.0;
    bool                                    enable_swing_lateral_accel_task_ = false;
    double                                  swing_lateral_accel_sign_ = 1.0;
    double                                  swing_lateral_accel_roll_kp_ = 0.0;
    double                                  swing_lateral_accel_roll_rate_kd_ = 0.0;
    double                                  swing_lateral_accel_momentum_scale_ = 0.0;
    double                                  swing_lateral_accel_max_ = 20.0;
    double                                  swing_lateral_accel_d_ = 0.0;

    bool                                    enable_roll_momentum_ref_ = true;
    double                                  roll_momentum_kp_ = 10.0;
    double                                  roll_momentum_kd_ = 2.0;
    double                                  roll_angle_momentum_kp_ = 0.0;
    double                                  roll_rate_momentum_kd_ = 0.0;
    double                                  roll_momentum_max_rate_ = 35.0;
    double                                  roll_momentum_sign_ = 1.0;
    bool                                    use_lateral_com_roll_momentum_ = false;
    Eigen::Vector3d                         roll_momentum_axis_ = Eigen::Vector3d::UnitX();
    double                                  roll_momentum_rate_d_ = 0.0;
    bool                                    enable_swing_leg_roll_momentum_task_ = true;
    double                                  swing_leg_roll_momentum_scale_ = 0.5;
    double                                  swing_leg_roll_momentum_max_rate_ = 300.0;

    bool                                    enable_single_wheel_stance_ = false;
    bool                                    enable_line_contact_wheel_control_ = true;
    bool                                    enable_single_wheel_pitch_control_ = true;
    double                                  single_wheel_pitch_d_ = 0.0;
    double                                  single_wheel_pitch_kp_ = 0.5;
    double                                  single_wheel_pitch_kd_ = 0.05;
    double                                  single_wheel_pitch_accel_kp_ = 0.0;
    double                                  single_wheel_pitch_accel_kd_ = 0.0;
    double                                  single_wheel_pitch_sign_ = 1.0;
    double                                  single_wheel_max_lin_vel_ = 0.25;
    double                                  single_wheel_max_lin_acc_ = 0.0;
    double                                  single_wheel_lin_vel_d_ = 0.0;
    double                                  single_wheel_lin_acc_d_ = 0.0;
    bool                                    enable_single_wheel_com_feedback_ = true;
    double                                  single_wheel_com_pitch_kp_ = 0.25;
    double                                  single_wheel_com_pitch_kd_ = 0.04;
    double                                  single_wheel_com_feedback_sign_ = 1.0;
    double                                  single_wheel_com_max_offset_ = 0.08;
    double                                  single_wheel_lateral_contact_kd_ = 20.0;
    double                                  single_wheel_sagittal_contact_kd_ = 20.0;
    double                                  line_contact_com_xy_force_mask_ = 0.2;
    double                                  line_contact_moment_xy_mask_ = 0.0;
    double                                  line_contact_orientation_xy_mask_ = 0.25;
    double                                  line_contact_yaw_mask_ = 0.1;
    bool                                    enable_line_contact_roll_com_feedback_ = false;
    double                                  line_contact_roll_com_sign_ = 1.0;
    double                                  line_contact_roll_com_kp_ = 0.0;
    double                                  line_contact_roll_com_kd_ = 0.0;
    double                                  line_contact_roll_com_max_offset_ = 0.0;
    double                                  unicycle_visual_wheel_phase_ = 0.0;

    void updateSwayReference();
    void updateCoMShiftReference();
    void updateRightFootLiftReference();
    void updateSwingLegReactionReference();
    void updateRollMomentumReference();
    void updateSingleWheelPitchCommand();
    void updateSingleWheelCoMReference();
    bool isRightFootLiftPhase() const;
    bool isSingleWheelStancePhase() const;
    bool isLineContactWheelControlPhase() const;
    double rightFootLiftStartTime() const;
    Eigen::Vector3d leftSupportPoint() const;
    void computeFourContactWBC();
    void updateModel();
    void updateCommand();
    void updateVisualization();
    void updateUnicycleReferenceVisualization();
    void readConfig(std::string config_file);

    //////////////// ARBML ////////////////
    Eigen::Vector3d                         p_lnk[NO_OF_BODY];
    Eigen::Matrix3d                         R_lnk[NO_OF_BODY];

    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jp_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jr_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     J_lnkCoM[NO_OF_BODY];

    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdotp_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdotr_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdot_lnkCoM[NO_OF_BODY];

    int                                     no_of_EE;
    vector<int>                             id_body_EE;
    vector<Eigen::Vector3d>                 p0_lnk2EE;
    vector<Eigen::Matrix3d>                 R0_lnk2EE;

    vector<Eigen::Vector3d>                 p_EE;
    vector<Eigen::Vector3d>                 pdot_EE;
    vector<Eigen::Vector3d>                 omega_EE;
    vector<Eigen::Matrix3d>                 R_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jp_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jr_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jdotp_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jdotr_EE;

    void initEEParameters(const mjModel* model);
    void computeLinkKinematics();
    void computeEEKinematics(Eigen::Matrix<double, mahru::nDoF, 1>& xidot);
    void computeContactKinematics();
    //////////////// ARBML ////////////////
};

#endif // __FSM_UNICYCLECTRL_HPP__
