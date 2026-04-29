#ifndef __FSM_BALANCECTRL_HPP__
#define __FSM_BALANCECTRL_HPP__

#include <string>
#include <mujoco/mujoco.h>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

#include "3rd-parties/ARBMLlib/ARBML.h"
#include "Controller/ConvexMPC/ConvexMpc.h"
#include "Controller/ConvexMPC/LocoCtrl.hpp"

using namespace mahru;

template <typename T>
class FSM_BalanceCtrlState : public States {
public:
    explicit FSM_BalanceCtrlState(RobotData& robot);
    ~FSM_BalanceCtrlState() { delete arbml_; }

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {};
    void runTransition() override {};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData*  robot_data_;
    CARBML*     arbml_ = nullptr;
    mujoco::TrajVizUtil* viz_ = nullptr;
    LocoCtrl loco_ctrl_;

    Eigen::Matrix<T, num_act_joint, 1>      jpos_0_;
    Eigen::Vector3d                         p_CoM_wbc_d_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d                         R_B_wbc_d_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d                         mpc_euler_d_ = Eigen::Vector3d::Zero();
    Eigen::Matrix<T, num_act_joint, 1>      torq_mpc_ = Eigen::Matrix<T, num_act_joint, 1>::Zero();
    Eigen::Matrix<double, ConvexMpc::kForceDim, 1> grf_mpc_ =
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();
    double                                  mpc_time_ = 0.0;
    double                                  mpc_dt_ = 0.05;
    bool                                    is_mpc_solved_ = false;
    
    void computeConvexMPC();
    void computeWeightedWBC();

    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);
    void applySwingTask();

    //////////////// ARBML ////////////////
	/////	Motion parameters for body frame : NOT NECESSARY !!!!
	///////////////////////////////////////////////////////////////////////////
	Eigen::Vector3d							p_lnk[NO_OF_BODY];			//	Position vector of i-th link frame in {I}
	Eigen::Matrix3d							R_lnk[NO_OF_BODY];			//	Rotation matrix of i-th link frame in {I}
    
	Eigen::Matrix<T, DOF3, mahru::nDoF>		Jp_lnk[NO_OF_BODY];			//	Linear Jacobian of i-th link in {I}
	Eigen::Matrix<T, DOF3, mahru::nDoF>		Jr_lnk[NO_OF_BODY];			//	Angular Jacobian of i-th link in {I}
	Eigen::Matrix<T, DOF3, mahru::nDoF>		J_lnkCoM[NO_OF_BODY];		//	CoM Jacobian of i-th link in {I}
    
	Eigen::Matrix<T, DOF3, mahru::nDoF>		Jdotp_lnk[NO_OF_BODY];		//	Time derivative of Jp_lnk
	Eigen::Matrix<T, DOF3, mahru::nDoF>		Jdotr_lnk[NO_OF_BODY];		//	Time derivative of Jr_lnk
	Eigen::Matrix<T, DOF3, mahru::nDoF>		Jdot_lnkCoM[NO_OF_BODY];	//	Time derivative of Jp_lnkCoM
    
	///////////////////////////////////////////////////////////////////////////
	/////	Motion parameters for end-effectors : Expressed in {I}
	///////////////////////////////////////////////////////////////////////////
	int												no_of_EE;			//	Number of end-effectors
	vector<int>										id_body_EE;			//	End-effector ID
	vector<Eigen::Vector3d>							p0_lnk2EE;			//	Local position offset from link frame to end-effector
	vector<Eigen::Matrix3d>							R0_lnk2EE;			//	Local rotation offset from link frame to end-effector
    
	vector<Eigen::Vector3d>							p_EE;				//	Position vector of i-th end-effector
	vector<Eigen::Vector3d>							pdot_EE;			//	Linear velocity of i-th end-effector
	vector<Eigen::Vector3d>							omega_EE;			//	Angular velocity of i-th end-effector
	vector<Eigen::Matrix3d>							R_EE;				//	End-effector rotation matrix
	vector<Eigen::Matrix<T, DOF3, mahru::nDoF>>		Jp_EE;				//	i-th End-effector linear Jacobian
	vector<Eigen::Matrix<T, DOF3, mahru::nDoF>>		Jr_EE;				//	i-th End-effector angular Jacobian
	vector<Eigen::Matrix<T, DOF3, mahru::nDoF>>		Jdotp_EE;			//	Time derivative of Jp_EE
	vector<Eigen::Matrix<T, DOF3, mahru::nDoF>>		Jdotr_EE;			//	Time derivative of Jr_EE
    
    void initEEParameters(const mjModel* model);
    void computeLinkKinematics();
    void computeEEKinematics(Eigen::Matrix<double, mahru::nDoF, 1>& xidot);
    void computeContactKinematics();
    //////////////// ARBML ////////////////
};

#endif // __FSM_BALANCECTRL_HPP__
