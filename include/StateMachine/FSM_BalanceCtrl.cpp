#include "FSM_BalanceCtrl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

template <typename T>
FSM_BalanceCtrlState<T>::FSM_BalanceCtrlState(RobotData& robot) :
    robot_data_(&robot)
{
    std::cout << "[ FSM_BalanceCtrlState ] Constructed" << std::endl;
    arbml_ = new CARBML();
}

template <typename T>
void FSM_BalanceCtrlState<T>::onEnter()
{
    std::cout << "[ FSM_BalanceCtrlState ] OnEnter" << std::endl;

    if (viz_) {
        viz_->clear();
    }

    std::cout << "[ FSM_BalanceCtrlState ] LoadModel" << std::endl;
	mjModel* mnew = mj_loadXML(
        (std::string(CMAKE_SOURCE_DIR) + std::string(mahru::model_xml)).c_str(), nullptr, nullptr, 0
    );
    arbml_->initRobot(mnew);
    initEEParameters(mnew);
    updateModel();

    // Initialize Command
    jpos_0_ = robot_data_->ctrl.jpos_d;
    p_CoM_wbc_d_ = arbml_->p_CoM;
    R_B_wbc_d_ = arbml_->R_B;
    const Eigen::Vector3d eul_zxy_d = _Rot2EulZXY(R_B_wbc_d_);
    mpc_euler_d_ << eul_zxy_d(1), eul_zxy_d(2), eul_zxy_d(0);
    torq_mpc_.setZero();
    grf_mpc_.setZero();
    mpc_time_ = 0.0;

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_BalanceCtrl_config.yaml");
    this->state_time = 0.0;
}

template <typename T>
void FSM_BalanceCtrlState<T>::runNominal()
{
    updateModel();
    computeConvexMPC();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}

template <typename T>
void FSM_BalanceCtrlState<T>::computeConvexMPC()
{
    constexpr double kControlDt = 0.001;

    if (mpc_time_ <= 0.0) {
        ConvexMpc::Input mpc_input;
        mpc_input.mass = arbml_->getTotalMass();
        mpc_input.R_B = arbml_->R_B;

        const Eigen::Vector3d eul_zxy = _Rot2EulZXY(arbml_->R_B);
        mpc_input.euler_B << eul_zxy(1), eul_zxy(2), eul_zxy(0);

        mpc_euler_d_(2) += robot_data_->ctrl.ang_vel_d(2) * mpc_dt_;
        mpc_input.euler_B_d = mpc_euler_d_;
        mpc_input.p_CoM = arbml_->p_CoM;
        mpc_input.pdot_CoM = arbml_->pdot_CoM;
        mpc_input.omega_B = arbml_->omega_B;
        mpc_input.lin_vel_d = robot_data_->ctrl.lin_vel_d;
        mpc_input.ang_vel_d = robot_data_->ctrl.ang_vel_d;
        mpc_input.com_height_d = p_CoM_wbc_d_(2);

        Eigen::Matrix3d I_G = Eigen::Matrix3d::Zero();
        for (int i = 0; i < static_cast<int>(mahru::NO_OF_BODY); ++i) {
            I_G += arbml_->I_G_BCS[i]
                 - arbml_->body[i].get_mass()
                   * Skew(arbml_->rpos_lnk[i]) * Skew(arbml_->rpos_lnk[i]);
        }
        mpc_input.trunk_inertia = I_G;

        for (int i = 0; i < ConvexMpc::kNumContacts; ++i) {
            mpc_input.contact_pos_abs.col(i) = robot_data_->fbk.p_C[i] - arbml_->p_CoM;
        }

        is_mpc_solved_ = convex_mpc_.update(mpc_input, mpc_dt_);
        if (is_mpc_solved_) {
            grf_mpc_ = convex_mpc_.groundReactionForce();
        }
    }

    Eigen::MatrixXd J_contact = Eigen::MatrixXd::Zero(ConvexMpc::kForceDim, mahru::nDoF);
    for (int i = 0; i < ConvexMpc::kNumContacts; ++i) {
        J_contact.block(DOF3 * i, 0, DOF3, mahru::nDoF) = robot_data_->fbk.Jp_C[i];
    }

    const Eigen::Matrix<double, mahru::nDoF, 1> generalized_torque =
        arbml_->C_mat * arbml_->xidot + arbml_->g_vec - J_contact.transpose() * grf_mpc_;

    torq_mpc_ = generalized_torque.tail(mahru::num_act_joint);
    torq_mpc_ += robot_data_->param.Kd.asDiagonal() * (-robot_data_->fbk.jvel);
    // torq_mpc_ += robot_data_->param.Kp.asDiagonal() * (jpos_0_ - robot_data_->fbk.jpos) \
        + robot_data_->param.Kd.asDiagonal() * (-robot_data_->fbk.jvel);

    for (int i = 0; i < torq_mpc_.size(); ++i) {
        if (!std::isfinite(torq_mpc_(i))) {
            torq_mpc_(i) = robot_data_->ctrl.torq_d(i);
        }
    }

    mpc_time_ += kControlDt;
    if (mpc_time_ > mpc_dt_) {
        mpc_time_ = 0.0;
    }
}

template <typename T>
void FSM_BalanceCtrlState<T>::updateModel()
{
    /////	01. Update Feedback Information
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

    arbml_->xiddot = (arbml_->xidot - arbml_->xidot_tmp) / 0.001;
    arbml_->xidot_tmp = arbml_->xidot;

    /////	02. Compute Kinematic Motion Core (w.r.t Base frame) - Called BEFORE All others !!!
    arbml_->computeMotionCore();
    /////	03. Compute CoM Kinematics : "computeMotionCore()" is followed by "computeCoMKinematics()"
    arbml_->computeCoMKinematics();
    /////	Compute Dynamics : "computeCoMKinematics()" is followes by "computeDynamics()"
    arbml_->computeDynamics();
    /////	Compute Link/CoM Kinematics (w.r.t Inertial frame) 
    computeLinkKinematics();
    /////	Compute End-effector Kinematics : "computeMotionCore()" is followed by "computeEEKinematics()"
    computeEEKinematics(arbml_->xidot);
    /////	Compute Contact Kinematics
    computeContactKinematics();
}

template <typename T>
void FSM_BalanceCtrlState<T>::updateCommand()
{
    robot_data_->ctrl.jpos_d = jpos_0_;
    robot_data_->ctrl.jvel_d.setZero();
    robot_data_->ctrl.torq_d = torq_mpc_;
}

template <typename T>
void FSM_BalanceCtrlState<T>::initEEParameters(const mjModel* model)
{
	int i;
	Eigen::Vector3d temp_vec;
	Eigen::Vector4d temp_quat;

	//////////	Get body ID for end-effectors (defined in XML file via model->site !)
	id_body_EE.clear();
	p0_lnk2EE.clear();
	R0_lnk2EE.clear();

	no_of_EE = model->nsite;
    std::cout << "[ FSM_BalanceCtrlState ] No of EE: " << no_of_EE << std::endl;
	for (i = 0; i < no_of_EE; i++) {
		id_body_EE.push_back(model->site_bodyid[i] - 1);

		temp_vec = {(sysReal)model->site_pos[i * 3],
					(sysReal)model->site_pos[i * 3 + 1],
					(sysReal)model->site_pos[i * 3 + 2]};
		p0_lnk2EE.push_back(temp_vec);				//	Set rel. position of end-effector ???

		temp_quat = {(sysReal)model->site_quat[i * 4],
					 (sysReal)model->site_quat[i * 4 + 1],
					 (sysReal)model->site_quat[i * 4 + 2],
					 (sysReal)model->site_quat[i * 4 + 3]};
		R0_lnk2EE.push_back(_Quat2Rot(temp_quat));	//	Set rel. orientation of end-effector
	}

	id_body_EE.shrink_to_fit();
	p0_lnk2EE.shrink_to_fit();
	R0_lnk2EE.shrink_to_fit();

	/////	Initialize transformation matrix about base, end-effector, contact wheel
	p_EE.resize(no_of_EE);
	R_EE.resize(no_of_EE);
	pdot_EE.resize(no_of_EE);
	omega_EE.resize(no_of_EE);

	Jp_EE.resize(no_of_EE);		//	Linear Jacobian of end-effectors
	Jr_EE.resize(no_of_EE);		//	Angular Jacobian of end-effectors
	Jdotp_EE.resize(no_of_EE);		//	Time derivative of Jp_EE
	Jdotr_EE.resize(no_of_EE);		//	Time derivative of Jr_EE

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
void FSM_BalanceCtrlState<T>::computeLinkKinematics()
{
    for (int i = 0; i < mahru::NO_OF_BODY; i++) {
        arbml_->getLinkPose(i, p_lnk[i], R_lnk[i]);
        arbml_->getBodyJacob(i, p_lnk[i], Jp_lnk[i], Jr_lnk[i]);
        arbml_->getBodyJacobDeriv(i, Jdotp_lnk[i], Jdotr_lnk[i]);
    }
}

template <typename T>
void FSM_BalanceCtrlState<T>::computeEEKinematics(Eigen::Matrix<double, mahru::nDoF, 1>& xidot)
{
	int i, j, k;

	/////	Position / Rotation matrix of end-effector w.r.t {I}
	for (i = 0; i < id_body_EE.size(); i++) {
		arbml_->getBodyPose(id_body_EE[i], p0_lnk2EE[i], R0_lnk2EE[i], p_EE[i], R_EE[i]);
	}

	/////	End-effector Jacobian & its time derivative w.r.t {I}  (Geometric Jacobian NOT analytic Jacobian !)
	for (i = 0; i < id_body_EE.size(); i++) {
		arbml_->getBodyJacob(id_body_EE[i], p_EE[i], Jp_EE[i], Jr_EE[i]);
		arbml_->getBodyJacobDeriv(id_body_EE[i], Jdotp_EE[i], Jdotr_EE[i]);
	}

	/////	Compute end-effector velocity expressed in {I}
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
void FSM_BalanceCtrlState<T>::computeContactKinematics()
{
	double radius_wheel = 0.075;
	double radius_Torus_sphere = 0;
	
	double radius_toe = 0.019;
	double radius_Torus_sphere_toe = 0;

	Eigen::Vector3d     normal_vec[num_leg];
	Eigen::Vector3d     wheel_sagittal_vec[num_leg];
	Eigen::Vector3d     wheel_tangent_vec[num_leg];
	Eigen::Vector3d     z_wheel_vec[num_leg];
	Eigen::Vector3d     wheel_d_k_vec[num_leg];
    
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

    robot_data_->fbk.p_C[0] = p_EE[2] - radius_toe * wheel_d_k_vec[0] - radius_Torus_sphere_toe * normal_vec[0]; // right toe
    robot_data_->fbk.R_C[0].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[0];
    robot_data_->fbk.R_C[0].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[0];
    robot_data_->fbk.R_C[0].block(0, Z_AXIS, 3, 1) = normal_vec[0];

    robot_data_->fbk.p_C[1] = p_EE[3] - radius_wheel * wheel_d_k_vec[0] - radius_Torus_sphere * normal_vec[0]; // right wheel
    robot_data_->fbk.R_C[1].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[0];
    robot_data_->fbk.R_C[1].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[0];
    robot_data_->fbk.R_C[1].block(0, Z_AXIS, 3, 1) = normal_vec[0];

    robot_data_->fbk.p_C[2] = p_EE[4] - radius_toe * wheel_d_k_vec[1] - radius_Torus_sphere_toe * normal_vec[1]; // left toe
    robot_data_->fbk.R_C[2].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[1];
    robot_data_->fbk.R_C[2].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[1];
    robot_data_->fbk.R_C[2].block(0, Z_AXIS, 3, 1) = normal_vec[1];

    robot_data_->fbk.p_C[3] = p_EE[5] - radius_wheel * wheel_d_k_vec[1] - radius_Torus_sphere * normal_vec[1]; // left wheel
    robot_data_->fbk.R_C[3].block(0, X_AXIS, 3, 1) = wheel_tangent_vec[1];
    robot_data_->fbk.R_C[3].block(0, Y_AXIS, 3, 1) = wheel_sagittal_vec[1];
    robot_data_->fbk.R_C[3].block(0, Z_AXIS, 3, 1) = normal_vec[1];

    for(int i=0; i<4; i++) {
        arbml_->getBodyJacob(id_body_EE[i+2], robot_data_->fbk.p_C[i], robot_data_->fbk.Jp_C[i], robot_data_->fbk.Jr_C[i]);
        arbml_->getBodyJacobDeriv(id_body_EE[i+2], robot_data_->fbk.Jdotp_C[i], robot_data_->fbk.Jdotr_C[i]);
    }

    // for(int i=0; i<4; i++) {
    //     const auto Jp_C = robot_data_->fbk.Jp_C[i];
    //     const auto Jr_C = robot_data_->fbk.Jr_C[i];
    //     robot_data_->fbk.Jdotp_C[i] = (Jp_C - robot_data_->fbk.Jp_C_prev[i]) / arbml_->getSamplingTime();
    //     robot_data_->fbk.Jdotr_C[i] = (Jr_C - robot_data_->fbk.Jr_C_prev[i]) / arbml_->getSamplingTime();
    //     robot_data_->fbk.Jp_C_prev[i] = Jp_C;
    //     robot_data_->fbk.Jr_C_prev[i] = Jr_C;
    // }

    robot_data_->fbk.pdot_C[0] = robot_data_->fbk.Jp_C[0] * arbml_->xidot;
    robot_data_->fbk.pdot_C[1] = robot_data_->fbk.Jp_C[1] * arbml_->xidot;
    robot_data_->fbk.pdot_C[2] = robot_data_->fbk.Jp_C[2] * arbml_->xidot;
    robot_data_->fbk.pdot_C[3] = robot_data_->fbk.Jp_C[3] * arbml_->xidot;
}

template <typename T>
void FSM_BalanceCtrlState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("BalanceCtrl/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
    
    viz_->sphere("BalanceCtrl/CoM", arbml_->p_CoM,
        0.035, {1.0f, 0.0f, 0.0f, 0.5f}
    );
    // viz_->sphere("BalanceCtrl/CoM_d", p_CoM_wbc_d_,
    //     0.025, {0.1f, 0.8f, 1.0f, 0.8f}
    // );

    viz_->clearPrefix("BalanceCtrl/MPC_Horizon");
    if (is_mpc_solved_ && convex_mpc_.predictedStates().allFinite()) {
        const Eigen::VectorXd horizon_states = convex_mpc_.predictedStates();
        viz_->mpcHorizon("BalanceCtrl/MPC_Horizon/path",
            horizon_states,
            ConvexMpc::kPlanHorizon,
            ConvexMpc::kStateDim,
            3,
            4.0,
            {1.0f, 0.0f, 0.0f, 0.8f}
        );

        for (int i = 0; i < ConvexMpc::kPlanHorizon; ++i) {
            const Eigen::Vector3d pos =
                convex_mpc_.predictedStates().segment<3>(i * ConvexMpc::kStateDim + 3);
            viz_->sphere("BalanceCtrl/MPC_Horizon/point_" + std::to_string(i),
                pos,
                0.012,
                {1.0f, 0.0f, 0.0f, 1.0f}
            );
        }
    }

    // viz_->pushTrail("BalanceCtrl/CoM_d_traj",
    //     p_CoM_wbc_d_,
    //     500, 4.0, {1.0f, 0.0f, 0.0f, 0.8f}
    // );
    viz_->pushTrail("BalanceCtrl/CoM_traj",
        arbml_->p_CoM,
        500, 4.0, {1.0f, 0.5f, 0.5f, 1.0f}
    );

    Eigen::Vector3d zmp_pos = arbml_->pos_ZMP;
    zmp_pos.z() = 0.0;
    viz_->cylinder("BalanceCtrl/ZMP", zmp_pos,
        0.03, 0.01, {1.0f, 0.9f, 0.0f, 0.8f}
    );
    viz_->label("BalanceCtrl/ZMP_label", zmp_pos,
        "ZMP", {1.0f, 0.9f, 0.0f, 1.0f}
    );

    // for(int i = 0; i < id_body_EE.size(); i++) {
    //     viz_->sphere("BalanceCtrl/EE" + std::to_string(i) + "_pos",
    //         p_EE[i], 0.02, {0.0f, 0.3f, 0.3f, 0.8f}
    //     );
    // }

    for(int i = 0; i < 4; i++) {
        viz_->sphere("BalanceCtrl/Contact_" + std::to_string(i) + "_pos",
            robot_data_->fbk.p_C[i], 0.02, {0.0f, 0.3f, 0.3f, 0.8f}
        );
    }
}

template <typename T>
void FSM_BalanceCtrlState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_BalanceCtrlState<T>::readConfig(std::string config_file)
{
    std::cout << "[ FSM_BalanceCtrlState ] readConfig: " << config_file << std::endl;
    const YAML::Node config = YAML::LoadFile(config_file);
    std::string path = config_file;
    YAML::Node yaml_node = YAML::LoadFile(path.c_str());
    try {
        for(int i = 0; i < num_act_joint; i++) {
            robot_data_->param.Kp[i] = yaml_node["Kp"][i].as<T>();
            robot_data_->param.Kd[i] = yaml_node["Kd"][i].as<T>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading target_jpos from config file: " << e.what() << std::endl;
    }
}

// template class FSM_BalanceCtrlState<float>;
template class FSM_BalanceCtrlState<double>;
