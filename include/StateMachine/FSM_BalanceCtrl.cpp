#include "FSM_BalanceCtrl.hpp"

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

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_BalanceCtrl_config.yaml");
    this->state_time = 0.0;
}

template <typename T>
void FSM_BalanceCtrlState<T>::runNominal()
{
    updateModel();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}

template <typename T>
void FSM_BalanceCtrlState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
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
}

template <typename T>
void FSM_BalanceCtrlState<T>::updateCommand()
{
    robot_data_->ctrl.jpos_d = jpos_0_;
    robot_data_->ctrl.jvel_d.setZero();
    robot_data_->ctrl.torq_d = robot_data_->param.Kp.asDiagonal() * (robot_data_->ctrl.jpos_d - robot_data_->fbk.jpos) + \
                                robot_data_->param.Kd.asDiagonal() * (robot_data_->ctrl.jvel_d - robot_data_->fbk.jvel);
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
	p_EE.reserve(no_of_EE);
	R_EE.reserve(no_of_EE);
	pdot_EE.reserve(no_of_EE);
	omega_EE.reserve(no_of_EE);

	Jp_EE.reserve(no_of_EE);		//	Linear Jacobian of end-effectors
	Jr_EE.reserve(no_of_EE);		//	Angular Jacobian of end-effectors
	Jdotp_EE.reserve(no_of_EE);		//	Time derivative of Jp_EE
	Jdotr_EE.reserve(no_of_EE);		//	Time derivative of Jr_EE

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
void FSM_BalanceCtrlState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("BalanceCtrl/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
    
    for(int i = 0; i < id_body_EE.size(); i++) {
        viz_->sphere("BalanceCtrl/EE" + std::to_string(i),
            p_EE[i],
            0.02, {0.0f, 0.3f, 0.3f, 0.8f}
        );
    }
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
