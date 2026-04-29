#include "FSM_UnicycleCtrl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kControlDt = 0.001;
constexpr double kPi = 3.14159265358979323846;
constexpr int kRightToeContact = 0;
constexpr int kRightWheelContact = 1;
constexpr int kLeftToeContact = 2;
constexpr int kLeftWheelContact = 3;
constexpr int kTorsoYawJoint = 0;
constexpr int kRightToeJoint = 12;
constexpr double kRightToeLiftTarget = 105.0 * kPi / 180.0;

double smoothstep(double x)
{
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

double smoothstepDerivative(double x)
{
    x = std::clamp(x, 0.0, 1.0);
    return 6.0 * x * (1.0 - x);
}

Eigen::Matrix3d yawRotation(const Eigen::Matrix3d& R_B)
{
    const double yaw = std::atan2(R_B(1, 0), R_B(0, 0));
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);

    Eigen::Matrix3d R_yaw = Eigen::Matrix3d::Identity();
    R_yaw(0, 0) = cos_yaw;
    R_yaw(0, 1) = -sin_yaw;
    R_yaw(1, 0) = sin_yaw;
    R_yaw(1, 1) = cos_yaw;
    return R_yaw;
}

}  // namespace

template <typename T>
FSM_UnicycleCtrlState<T>::FSM_UnicycleCtrlState(RobotData& robot) :
    robot_data_(&robot)
{
    std::cout << "[ FSM_UnicycleCtrlState ] Constructed" << std::endl;
    arbml_ = new CARBML();
}

template <typename T>
void FSM_UnicycleCtrlState<T>::onEnter()
{
    std::cout << "[ FSM_UnicycleCtrlState ] OnEnter" << std::endl;

    if (viz_) {
        // viz_->clearPrefix("UnicycleCtrl/");
        viz_->clear();
    }

    std::cout << "[ FSM_UnicycleCtrlState ] LoadModel" << std::endl;
    mjModel* mnew = mj_loadXML(
        (std::string(CMAKE_SOURCE_DIR) + std::string(mahru::model_xml)).c_str(), nullptr, nullptr, 0
    );
    if (mnew == nullptr) {
        std::cerr << "[ FSM_UnicycleCtrlState ] Failed to load MuJoCo model" << std::endl;
        return;
    }
    arbml_->initRobot(mnew);
    initEEParameters(mnew);
    updateModel();

    jpos_0_ = robot_data_->ctrl.jpos_d.template cast<T>();
    p_CoM_nominal_ = arbml_->p_CoM;
    p_CoM_wbc_d_ = p_CoM_nominal_;
    pdot_CoM_wbc_d_.setZero();
    pddot_CoM_wbc_ff_.setZero();
    R_B_wbc_d_ = arbml_->R_B;
    torq_wbc_.setZero();
    nominal_grf_.setZero();
    robot_data_->ctrl.contact_schedule.setOnes();
    robot_data_->ctrl.lin_vel_d.setZero();
    robot_data_->ctrl.ang_vel_d.setZero();
    com_shift_initialized_ = false;
    com_shift_start_.setZero();
    com_shift_target_.setZero();
    right_foot_lift_initialized_ = false;
    right_wheel_lift_start_.setZero();
    right_wheel_lift_ref_.setZero();
    right_wheel_lift_vel_.setZero();
    roll_momentum_axis_.setUnit(X_AXIS);
    roll_momentum_rate_d_ = 0.0;

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_UnicycleCtrl_config.yaml");
    this->state_time = 0.0;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::runNominal()
{
    updateModel();
    updateSwayReference();
    updateCoMShiftReference();
    updateRightFootLiftReference();
    updateRollMomentumReference();
    computeFourContactWBC();
    updateCommand();
    updateVisualization();

    this->state_time += kControlDt;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateSwayReference()
{
    p_CoM_wbc_d_ = p_CoM_nominal_;
    pdot_CoM_wbc_d_.setZero();
    pddot_CoM_wbc_ff_.setZero();

    if (!enable_sway_) {
        return;
    }

    const double ramp =
        sway_ramp_time_ > 1e-6 ? smoothstep(this->state_time / sway_ramp_time_) : 1.0;
    const double omega = 2.0 * kPi * sway_frequency_hz_;
    const double phase = omega * this->state_time;
    const double sin_phase = std::sin(phase);
    const double cos_phase = std::cos(phase);

    p_CoM_wbc_d_ += ramp * sway_amplitude_ * sin_phase;
    pdot_CoM_wbc_d_ = ramp * sway_amplitude_ * omega * cos_phase;
    pddot_CoM_wbc_ff_ = -ramp * sway_amplitude_ * omega * omega * sin_phase;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateCoMShiftReference()
{
    if (!enable_com_shift_) {
        return;
    }

    if (!com_shift_initialized_) {
        com_shift_start_ = arbml_->p_CoM;
        com_shift_target_ = leftSupportPoint() + com_shift_offset_;
        com_shift_target_.z() = p_CoM_nominal_.z() + com_shift_offset_.z();
        com_shift_initialized_ = true;
    }

    const double duration = std::max(com_shift_duration_, kControlDt);
    const double phase = this->state_time / duration;
    const double s = smoothstep(phase);
    const double sdot = smoothstepDerivative(phase) / duration;
    const Eigen::Vector3d delta = com_shift_target_ - com_shift_start_;

    p_CoM_wbc_d_ = com_shift_start_ + s * delta;
    pdot_CoM_wbc_d_ = sdot * delta;
    pddot_CoM_wbc_ff_.setZero();

    if (phase >= 1.0) {
        p_CoM_wbc_d_ = com_shift_target_;
        pdot_CoM_wbc_d_.setZero();
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateRightFootLiftReference()
{
    if (!isRightFootLiftPhase()) {
        return;
    }

    const Eigen::Vector3d p_pelvis = robot_data_->fbk.p_B;
    const Eigen::Matrix3d R_yaw = yawRotation(robot_data_->fbk.R_B);

    if (!right_foot_lift_initialized_) {
        right_wheel_lift_start_ = robot_data_->fbk.p_C[kRightWheelContact];
        right_wheel_lift_ref_ = right_wheel_lift_start_;
        right_wheel_lift_vel_.setZero();
        if (!right_wheel_lift_pelvis_offset_configured_) {
            right_wheel_lift_pelvis_offset_ =
                R_yaw.transpose() * (right_wheel_lift_start_ - p_pelvis);
            right_wheel_lift_pelvis_offset_.z() += right_foot_lift_height_;
        }
        right_foot_lift_initialized_ = true;
    }

    const double lift_time = this->state_time - rightFootLiftStartTime();
    const double duration = std::max(right_foot_lift_duration_, kControlDt);
    const double phase = lift_time / duration;
    const double s = smoothstep(phase);
    const double sdot = smoothstepDerivative(phase) / duration;

    const Eigen::Vector3d yaw_offset = R_yaw * right_wheel_lift_pelvis_offset_;
    const Eigen::Vector3d pelvis_frame_target = p_pelvis + yaw_offset;
    const double yaw_rate = robot_data_->fbk.omega_B.z();
    Eigen::Vector3d pelvis_frame_target_vel = robot_data_->fbk.pdot_B;
    pelvis_frame_target_vel.x() += -yaw_rate * yaw_offset.y();
    pelvis_frame_target_vel.y() += yaw_rate * yaw_offset.x();

    right_wheel_lift_ref_ =
        (1.0 - s) * right_wheel_lift_start_ + s * pelvis_frame_target;
    right_wheel_lift_vel_ =
        sdot * (pelvis_frame_target - right_wheel_lift_start_)
        + s * pelvis_frame_target_vel;

    if (phase >= 1.0) {
        right_wheel_lift_ref_ = pelvis_frame_target;
        right_wheel_lift_vel_ = pelvis_frame_target_vel;
    }
}

template <typename T>
bool FSM_UnicycleCtrlState<T>::isRightFootLiftPhase() const
{
    return enable_right_foot_lift_
        && this->state_time >= rightFootLiftStartTime();
}

template <typename T>
double FSM_UnicycleCtrlState<T>::rightFootLiftStartTime() const
{
    if (!enable_com_shift_) {
        return right_foot_lift_start_time_;
    }
    return std::max(right_foot_lift_start_time_, com_shift_duration_);
}

template <typename T>
Eigen::Vector3d FSM_UnicycleCtrlState<T>::leftSupportPoint() const
{
    return 0.5 * (
        robot_data_->fbk.p_C[kLeftToeContact]
        + robot_data_->fbk.p_C[kLeftWheelContact]);
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateRollMomentumReference()
{
    roll_momentum_axis_.setUnit(X_AXIS);
    roll_momentum_rate_d_ = 0.0;
    robot_data_->ctrl.unicycle_state_time = this->state_time;
    robot_data_->ctrl.right_foot_lift_phase = isRightFootLiftPhase() ? 1.0 : 0.0;
    robot_data_->ctrl.roll_momentum_rate_d = 0.0;
    robot_data_->ctrl.roll_momentum_rate_actual = 0.0;
    robot_data_->ctrl.roll_momentum_y_err = 0.0;
    robot_data_->ctrl.roll_momentum_ydot_err = 0.0;
    robot_data_->ctrl.roll_momentum_height = 0.0;

    Eigen::Vector3d roll_axis = robot_data_->fbk.R_B.col(X_AXIS);
    roll_axis.z() = 0.0;
    if (roll_axis.norm() < 1e-6) {
        roll_axis.setUnit(X_AXIS);
    } else {
        roll_axis.normalize();
    }

    const double roll_momentum_rate_actual =
        (roll_axis.transpose() * arbml_->kdot_CoM)(0);
    if (std::isfinite(roll_momentum_rate_actual)) {
        robot_data_->ctrl.roll_momentum_rate_actual = roll_momentum_rate_actual;
    }

    if (!enable_roll_momentum_ref_ || !isRightFootLiftPhase()) {
        return;
    }

    Eigen::Vector3d lateral_axis = robot_data_->fbk.R_B.col(Y_AXIS);
    lateral_axis.z() = 0.0;
    if (lateral_axis.norm() < 1e-6) {
        lateral_axis.setUnit(Y_AXIS);
    } else {
        lateral_axis.normalize();
    }

    const Eigen::Vector3d p_support = leftSupportPoint();
    const Eigen::Vector3d p_com =
        robot_data_->fbk.p_CoM.z() > 1e-6 ? robot_data_->fbk.p_CoM : arbml_->p_CoM;
    const Eigen::Vector3d pdot_com =
        robot_data_->fbk.p_CoM.z() > 1e-6 ? robot_data_->fbk.pdot_CoM : arbml_->pdot_CoM;
    const double y_err = lateral_axis.dot(p_com - p_support);
    const double ydot_err = lateral_axis.dot(pdot_com);
    const double yddot_d =
        -roll_momentum_kp_ * y_err - roll_momentum_kd_ * ydot_err;
    const double height = std::max(0.2, p_com.z() - p_support.z());
    const double mass = arbml_->getTotalMass();
    const double gravity = arbml_->getGravityConst();

    roll_momentum_axis_ = roll_axis;
    roll_momentum_rate_d_ =
        roll_momentum_sign_ * mass * (height * yddot_d - gravity * y_err);
    roll_momentum_rate_d_ = std::clamp(
        roll_momentum_rate_d_,
        -roll_momentum_max_rate_,
        roll_momentum_max_rate_);
    robot_data_->ctrl.roll_momentum_rate_d = roll_momentum_rate_d_;
    robot_data_->ctrl.roll_momentum_y_err = y_err;
    robot_data_->ctrl.roll_momentum_ydot_err = ydot_err;
    robot_data_->ctrl.roll_momentum_height = height;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateModel()
{
    arbml_->p_B = robot_data_->fbk.p_B;
    arbml_->quat_B(0) = robot_data_->fbk.quat_B.w();
    arbml_->quat_B(1) = robot_data_->fbk.quat_B.x();
    arbml_->quat_B(2) = robot_data_->fbk.quat_B.y();
    arbml_->quat_B(3) = robot_data_->fbk.quat_B.z();

    arbml_->R_B = robot_data_->fbk.R_B;
    arbml_->varphi_B = robot_data_->fbk.varphi_B;
    arbml_->omega_B = robot_data_->fbk.omega_B;

    arbml_->q = robot_data_->fbk.jpos;
    arbml_->qdot = robot_data_->fbk.jvel;

    arbml_->xi_quat.segment(0, 3) = robot_data_->fbk.p_B;
    arbml_->xi_quat.segment(3, 4) = arbml_->quat_B;
    arbml_->xi_quat.tail(mahru::num_act_joint) = robot_data_->fbk.jpos;

    arbml_->xidot.segment(0, 3) = robot_data_->fbk.pdot_B;
    arbml_->xidot.segment(3, 3) = robot_data_->fbk.omega_B;
    arbml_->xidot.tail(mahru::num_act_joint) = robot_data_->fbk.jvel;

    arbml_->xiddot = (arbml_->xidot - arbml_->xidot_tmp) / kControlDt;
    arbml_->xidot_tmp = arbml_->xidot;

    arbml_->computeMotionCore();
    arbml_->computeCoMKinematics();
    if (arbml_->p_CoM.allFinite() && arbml_->p_CoM.z() > 1e-6) {
        robot_data_->fbk.p_CoM = arbml_->p_CoM;
        robot_data_->fbk.pdot_CoM = arbml_->pdot_CoM;
    }
    arbml_->computeDynamics();
    computeLinkKinematics();
    computeEEKinematics(arbml_->xidot);
    computeContactKinematics();
}

template <typename T>
void FSM_UnicycleCtrlState<T>::computeFourContactWBC()
{
    robot_data_->ctrl.contact_schedule.setOnes();
    if (isRightFootLiftPhase()) {
        robot_data_->ctrl.contact_schedule.row(kRightToeContact).setZero();
        robot_data_->ctrl.contact_schedule.row(kRightWheelContact).setZero();
    }
    robot_data_->ctrl.lin_vel_d.setZero();
    robot_data_->ctrl.ang_vel_d.setZero();

    const Eigen::Matrix<T, mahru::num_act_joint, 1> fallback_torque =
        robot_data_->param.Kp.asDiagonal() * (jpos_0_ - robot_data_->fbk.jpos)
        + robot_data_->param.Kd.asDiagonal() * (-robot_data_->fbk.jvel);
    torq_wbc_ = fallback_torque;

    WeightedWBC::Input wbc_input;
    wbc_input.q_d = jpos_0_.template cast<double>();
    wbc_input.qdot_d.setZero();
    wbc_input.qddot_d.setZero();
    wbc_input.q_d(kTorsoYawJoint) = 0.0;
    wbc_input.p_CoM_d = p_CoM_wbc_d_;
    wbc_input.pdot_CoM_d = pdot_CoM_wbc_d_;
    wbc_input.pddot_CoM_ff = pddot_CoM_wbc_ff_;
    wbc_input.lin_vel_d.setZero();
    wbc_input.R_B_d = R_B_wbc_d_;
    wbc_input.grfs_mpc = nominal_grf_;
    wbc_input.swing_contact_index = -1;
    if (isRightFootLiftPhase()) {
        wbc_input.swing_contact_index = kRightWheelContact;
        wbc_input.p_sw_d = right_wheel_lift_ref_;
        wbc_input.pdot_sw_d = right_wheel_lift_vel_;
    }
    wbc_input.enable_centroidal_force_task = true;
    if (enable_roll_momentum_ref_ && isRightFootLiftPhase()) {
        wbc_input.enable_roll_angular_momentum_task = true;
        wbc_input.roll_angular_momentum_axis = roll_momentum_axis_;
        wbc_input.roll_angular_momentum_rate_d = roll_momentum_rate_d_;
    }

    const WeightedWBC::Output wbc_output =
        weighted_wbc_.update(*arbml_, *robot_data_, wbc_input);
    is_wbc_solved_ = wbc_output.solved && wbc_output.torq_ff.allFinite();
    if (is_wbc_solved_) {
        torq_wbc_ = wbc_output.torq_ff.template cast<T>();
    }
    if (isRightFootLiftPhase()) {
        torq_wbc_(kRightToeJoint) =
            robot_data_->param.Kp(kRightToeJoint)
            * (kRightToeLiftTarget - robot_data_->fbk.jpos(kRightToeJoint))
            - robot_data_->param.Kd(kRightToeJoint) * robot_data_->fbk.jvel(kRightToeJoint);
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateCommand()
{
    robot_data_->ctrl.jpos_d = jpos_0_;
    robot_data_->ctrl.jvel_d.setZero();
    robot_data_->ctrl.torq_d = torq_wbc_;
}

template <typename T>
void FSM_UnicycleCtrlState<T>::initEEParameters(const mjModel* model)
{
    int i;
    Eigen::Vector3d temp_vec;
    Eigen::Vector4d temp_quat;

    id_body_EE.clear();
    p0_lnk2EE.clear();
    R0_lnk2EE.clear();

    no_of_EE = model->nsite;
    std::cout << "[ FSM_UnicycleCtrlState ] No of EE: " << no_of_EE << std::endl;
    for (i = 0; i < no_of_EE; i++) {
        id_body_EE.push_back(model->site_bodyid[i] - 1);

        temp_vec = {(sysReal)model->site_pos[i * 3],
                    (sysReal)model->site_pos[i * 3 + 1],
                    (sysReal)model->site_pos[i * 3 + 2]};
        p0_lnk2EE.push_back(temp_vec);

        temp_quat = {(sysReal)model->site_quat[i * 4],
                     (sysReal)model->site_quat[i * 4 + 1],
                     (sysReal)model->site_quat[i * 4 + 2],
                     (sysReal)model->site_quat[i * 4 + 3]};
        R0_lnk2EE.push_back(_Quat2Rot(temp_quat));
    }

    id_body_EE.shrink_to_fit();
    p0_lnk2EE.shrink_to_fit();
    R0_lnk2EE.shrink_to_fit();

    p_EE.resize(no_of_EE);
    R_EE.resize(no_of_EE);
    pdot_EE.resize(no_of_EE);
    omega_EE.resize(no_of_EE);

    Jp_EE.resize(no_of_EE);
    Jr_EE.resize(no_of_EE);
    Jdotp_EE.resize(no_of_EE);
    Jdotr_EE.resize(no_of_EE);

    for (int i = 0; i < no_of_EE; i++) {
        p_EE[i].setZero();
        R_EE[i].setIdentity();
        pdot_EE[i].setZero();
        omega_EE[i].setZero();

        Jp_EE[i].setZero();
        Jr_EE[i].setZero();
        Jdotp_EE[i].setZero();
        Jdotr_EE[i].setZero();
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::computeLinkKinematics()
{
    for (int i = 0; i < mahru::NO_OF_BODY; i++) {
        arbml_->getLinkPose(i, p_lnk[i], R_lnk[i]);
        arbml_->getBodyJacob(i, p_lnk[i], Jp_lnk[i], Jr_lnk[i]);
        arbml_->getBodyJacobDeriv(i, Jdotp_lnk[i], Jdotr_lnk[i]);
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::computeEEKinematics(Eigen::Matrix<double, mahru::nDoF, 1>& xidot)
{
    int i, j, k;

    for (i = 0; i < id_body_EE.size(); i++) {
        arbml_->getBodyPose(id_body_EE[i], p0_lnk2EE[i], R0_lnk2EE[i], p_EE[i], R_EE[i]);
    }

    for (i = 0; i < id_body_EE.size(); i++) {
        arbml_->getBodyJacob(id_body_EE[i], p_EE[i], Jp_EE[i], Jr_EE[i]);
        arbml_->getBodyJacobDeriv(id_body_EE[i], Jdotp_EE[i], Jdotr_EE[i]);
    }

    for (i = 0; i < id_body_EE.size(); i++) {
        pdot_EE[i].setZero();
        omega_EE[i].setZero();

#ifdef _FLOATING_BASE
        for (j = 0; j < DOF3; j++) {
            for (k = 0; k < DOF6; k++) {
                pdot_EE[i](j) += Jp_EE[i](j, k) * xidot(k);
                omega_EE[i](j) += Jr_EE[i](j, k) * xidot(k);
            }
        }
#endif
        for (j = 0; j < DOF3; j++) {
            for (unsigned& idx : arbml_->kinematic_chain[id_body_EE[i]]) {
                k = idx + mahru::nDoF_base - arbml_->BodyID_ActJntStart();
                pdot_EE[i](j) += Jp_EE[i](j, k) * xidot(k);
                omega_EE[i](j) += Jr_EE[i](j, k) * xidot(k);
            }
        }
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::computeContactKinematics()
{
    double radius_wheel = 0.075;
    double radius_Torus_sphere = 0;

    double radius_toe = 0.019;
    double radius_Torus_sphere_toe = 0;

    Eigen::Vector3d normal_vec[num_leg];
    Eigen::Vector3d wheel_sagittal_vec[num_leg];
    Eigen::Vector3d wheel_tangent_vec[num_leg];
    Eigen::Vector3d z_wheel_vec[num_leg];
    Eigen::Vector3d wheel_d_k_vec[num_leg];

    normal_vec[0] = {0.0, 0.0, 1.0};
    z_wheel_vec[0] = R_EE[3].block(0, Z_AXIS, 3, 1);
    wheel_tangent_vec[0] = normalize(z_wheel_vec[0].cross(normal_vec[0]));
    wheel_sagittal_vec[0] = normalize(normal_vec[0].cross(wheel_tangent_vec[0]));
    wheel_d_k_vec[0] = normalize(wheel_tangent_vec[0].cross(z_wheel_vec[0]));

    normal_vec[1] = {0.0, 0.0, 1.0};
    z_wheel_vec[1] = R_EE[5].block(0, Z_AXIS, 3, 1);
    wheel_tangent_vec[1] = normalize(z_wheel_vec[1].cross(normal_vec[1]));
    wheel_sagittal_vec[1] = normalize(normal_vec[1].cross(wheel_tangent_vec[1]));
    wheel_d_k_vec[1] = normalize(wheel_tangent_vec[1].cross(z_wheel_vec[1]));

    robot_data_->fbk.p_C[0] =
        p_EE[2] - radius_toe * wheel_d_k_vec[0] - radius_Torus_sphere_toe * normal_vec[0];
    robot_data_->fbk.R_C[0].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[0];
    robot_data_->fbk.R_C[0].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[0];
    robot_data_->fbk.R_C[0].block(0, Z_AXIS, 3, 1) = normal_vec[0];

    robot_data_->fbk.p_C[1] =
        p_EE[3] - radius_wheel * wheel_d_k_vec[0] - radius_Torus_sphere * normal_vec[0];
    robot_data_->fbk.R_C[1].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[0];
    robot_data_->fbk.R_C[1].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[0];
    robot_data_->fbk.R_C[1].block(0, Z_AXIS, 3, 1) = normal_vec[0];

    robot_data_->fbk.p_C[2] =
        p_EE[4] - radius_toe * wheel_d_k_vec[1] - radius_Torus_sphere_toe * normal_vec[1];
    robot_data_->fbk.R_C[2].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[1];
    robot_data_->fbk.R_C[2].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[1];
    robot_data_->fbk.R_C[2].block(0, Z_AXIS, 3, 1) = normal_vec[1];

    robot_data_->fbk.p_C[3] =
        p_EE[5] - radius_wheel * wheel_d_k_vec[1] - radius_Torus_sphere * normal_vec[1];
    robot_data_->fbk.R_C[3].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[1];
    robot_data_->fbk.R_C[3].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[1];
    robot_data_->fbk.R_C[3].block(0, Z_AXIS, 3, 1) = normal_vec[1];

    for (int i = 0; i < 4; i++) {
        arbml_->getBodyJacob(
            id_body_EE[i + 2], robot_data_->fbk.p_C[i],
            robot_data_->fbk.Jp_C[i], robot_data_->fbk.Jr_C[i]);
        arbml_->getBodyJacobDeriv(
            id_body_EE[i + 2],
            robot_data_->fbk.Jdotp_C[i], robot_data_->fbk.Jdotr_C[i]);
    }

    for (int i = 0; i < 4; i++) {
        robot_data_->fbk.pdot_C[i] = robot_data_->fbk.Jp_C[i] * arbml_->xidot;
    }
}

template <typename T>
void FSM_UnicycleCtrlState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("UnicycleCtrl/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
    viz_->frame("UnicycleCtrl/base_frame",
        robot_data_->fbk.p_B,
        robot_data_->fbk.R_B,
        0.1, 0.005
    );
    viz_->sphere("UnicycleCtrl/CoM",
        arbml_->p_CoM,
        0.035, {1.0f, 0.0f, 0.0f, 0.55f}
    );
    viz_->sphere("UnicycleCtrl/CoM_d",
        p_CoM_wbc_d_,
        0.025, {0.1f, 0.8f, 1.0f, 0.85f}
    );
    viz_->line("UnicycleCtrl/CoM_error",
        arbml_->p_CoM,
        p_CoM_wbc_d_,
        1.5, {0.1f, 0.8f, 1.0f, 0.65f}
    );
    viz_->pushTrail("UnicycleCtrl/CoM_d_trail",
        p_CoM_wbc_d_,
        500, 4.0, {0.1f, 0.8f, 1.0f, 0.75f}
    );
    viz_->pushTrail("UnicycleCtrl/CoM_trail",
        arbml_->p_CoM,
        500, 4.0, {1.0f, 0.5f, 0.5f, 1.0f}
    );

    for (int i = 0; i < 4; ++i) {
        viz_->sphere("UnicycleCtrl/Contact_" + std::to_string(i),
            robot_data_->fbk.p_C[i],
            0.018, {0.0f, 0.3f, 0.9f, 0.85f}
        );
    }
    if (com_shift_initialized_) {
        viz_->sphere("UnicycleCtrl/CoM_shift_target",
            com_shift_target_,
            0.022, {0.2f, 1.0f, 0.4f, 0.85f}
        );
    }
    if (right_foot_lift_initialized_) {
        viz_->sphere("UnicycleCtrl/RightWheel_lift_ref",
            right_wheel_lift_ref_,
            0.02, {1.0f, 0.2f, 0.2f, 0.85f}
        );
        viz_->line("UnicycleCtrl/RightWheel_lift_error",
            robot_data_->fbk.p_C[kRightWheelContact],
            right_wheel_lift_ref_,
            1.5, {1.0f, 0.2f, 0.2f, 0.65f}
        );
    }

    Eigen::Vector3d zmp_pos = arbml_->pos_ZMP;
    zmp_pos.z() = 0.0;
    viz_->cylinder("UnicycleCtrl/ZMP", zmp_pos,
        0.03, 0.01, {1.0f, 0.9f, 0.0f, 0.8f}
    );
}

template <typename T>
void FSM_UnicycleCtrlState<T>::readConfig(std::string config_file)
{
    std::cout << "[ FSM_UnicycleCtrlState ] readConfig: " << config_file << std::endl;
    const YAML::Node yaml_node = YAML::LoadFile(config_file);
    try {
        if (yaml_node["Kp"]) {
            for (int i = 0; i < num_act_joint; i++) {
                robot_data_->param.Kp[i] = yaml_node["Kp"][i].as<T>();
            }
        }
        if (yaml_node["Kd"]) {
            for (int i = 0; i < num_act_joint; i++) {
                robot_data_->param.Kd[i] = yaml_node["Kd"][i].as<T>();
            }
        }
        if (yaml_node["sway"]) {
            const YAML::Node sway = yaml_node["sway"];
            if (sway["enabled"]) {
                enable_sway_ = sway["enabled"].as<bool>();
            }
            if (sway["amplitude_m"]) {
                for (int i = 0; i < DOF3; ++i) {
                    sway_amplitude_(i) = sway["amplitude_m"][i].as<double>();
                }
            }
            if (sway["frequency_hz"]) {
                sway_frequency_hz_ = sway["frequency_hz"].as<double>();
            }
            if (sway["ramp_time"]) {
                sway_ramp_time_ = sway["ramp_time"].as<double>();
            }
        }
        if (yaml_node["com_shift"]) {
            const YAML::Node shift = yaml_node["com_shift"];
            if (shift["enabled"]) {
                enable_com_shift_ = shift["enabled"].as<bool>();
            }
            if (shift["duration"]) {
                com_shift_duration_ = shift["duration"].as<double>();
            }
            if (shift["offset_m"]) {
                for (int i = 0; i < DOF3; ++i) {
                    com_shift_offset_(i) = shift["offset_m"][i].as<double>();
                }
            }
        }
        if (yaml_node["right_foot_lift"]) {
            const YAML::Node lift = yaml_node["right_foot_lift"];
            if (lift["enabled"]) {
                enable_right_foot_lift_ = lift["enabled"].as<bool>();
            }
            if (lift["start_time"]) {
                right_foot_lift_start_time_ = lift["start_time"].as<double>();
            }
            if (lift["lift_height"]) {
                right_foot_lift_height_ = lift["lift_height"].as<double>();
            }
            if (lift["lift_duration"]) {
                right_foot_lift_duration_ = lift["lift_duration"].as<double>();
            }
            if (lift["pelvis_offset_m"]) {
                for (int i = 0; i < DOF3; ++i) {
                    right_wheel_lift_pelvis_offset_(i) =
                        lift["pelvis_offset_m"][i].as<double>();
                }
                right_wheel_lift_pelvis_offset_configured_ = true;
            }
        }
        if (yaml_node["roll_angular_momentum"]) {
            const YAML::Node roll = yaml_node["roll_angular_momentum"];
            if (roll["enabled"]) {
                enable_roll_momentum_ref_ = roll["enabled"].as<bool>();
            }
            if (roll["kp"]) {
                roll_momentum_kp_ = roll["kp"].as<double>();
            }
            if (roll["kd"]) {
                roll_momentum_kd_ = roll["kd"].as<double>();
            }
            if (roll["max_rate"]) {
                roll_momentum_max_rate_ = roll["max_rate"].as<double>();
            }
            if (roll["sign"]) {
                roll_momentum_sign_ = roll["sign"].as<double>();
            }
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading UnicycleCtrl config file: " << e.what() << std::endl;
    }
}

// template class FSM_UnicycleCtrlState<float>;
template class FSM_UnicycleCtrlState<double>;
