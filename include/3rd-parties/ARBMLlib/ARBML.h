//
#pragma once

// #include "RobotConfig.h"
#include "RobotDefinition.hpp"
#include "RigidBody.h"
#include <vector>
#include <chrono>
#include <mujoco/mujoco.h>

using namespace std;
using namespace mahru;

class CARBML {
private:
	int			_actJnt_start_bodyID;	//	Starting Body ID of active joint
	int			_FloatingBaseFlag;		//	Flag for floating-base body : 1 = floating-base, 0 = fixed-base !
	sysReal		_del_T;					//	Sampling Time : _del_T = mjModel->opt.timestep
	sysReal		mass_G;					//	Total mass of the system
	sysReal		_g_const;				//	9.81 !

	Eigen::Vector3d			e1, e2, e3;	//	Selection vector : e1 = (1, 0, 0), e2 = (0, 1, 0), e3 = (0, 0, 1)
	Eigen::Matrix3d	E3;			//	3 x 3 Identity matrix								

	/////	Temporal vectors & matrices
	Eigen::Vector3d				tempVec3;
	Eigen::Matrix3d		tempMat3x3, Tmp_D;
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> tempMat3xAct_1, tempMat3xAct_2;

public:
	int		IsDynamicsDone;

	CRigidBody body[mahru::NO_OF_BODY];	//	All moving bodies(active/passive) = mjModel->nbody - 1

	///////////////////////////////////////////////////////////////////////////
	/////	Structural information for URDF/MJCF parsing & initialization
	///////////////////////////////////////////////////////////////////////////
	vector<int>			actID;
	vector<int>			id_body;				//	ID of the body including fixed bodies
	vector<int>			joint_dir;				//	Joint axis direction : X = 0, Y = 1, Z = 2
	vector<int>			joint_type;				//	Joint type : Free -> Ball -> Prismatic -> Revolute !!!!
	vector<int>			id_body_parent;			//	ID of the parents body

	vector<int>			id_limited_joint;		//	ID of joint with limited joint range
	vector<sysReal>		q_min, q_max;			//	Range of joint position

	vector<sysReal>					mass_lnk;		//	Link mass
	vector<Eigen::Vector3d>			rho_LCS;		//	Local CoM position from its body/link frame : const
	vector<Eigen::Vector3d>			joint_axis;		//	Local joint axis vector : e1, e2, e3 !!!
	vector<Eigen::Vector3d>			pos0_offset;	//	Local position offset {i}-frame rel. to parent frame : const
	vector<Eigen::Matrix3d>			Rot0_offset;	//	Local rotation offset {i}-frame rel. to parent frame : const
	vector<Eigen::Matrix3d>			I_CoM_diag;		//	The principal moments of inertia matrix@CoM (diagonal)
	vector<Eigen::Matrix3d>			Rot_LCS2CoM;	//	Const. rotation matrix btw link frame and principal axis frame

	vector<vector<unsigned>>		kinematic_chain;//	Description of kinematic chain for bodies from base body 0/1.


	//////////////////////////////////////////////////////////////////////
	/////	Base frame-based motion parameters : Expressed in {B}
	//////////////////////////////////////////////////////////////////////
	Eigen::Vector3d					jntAxis_BCS[mahru::NO_OF_BODY];		//	Joint axis in {B}
	Eigen::Vector3d					jntAxisdot_BCS[mahru::NO_OF_BODY];		//	Time deriv. of jntAixs_BCS

	Eigen::Vector3d					rpos_lnk[mahru::NO_OF_BODY];			//	Position of i-th link w.r.t {B} expressed in {B}
	Eigen::Vector3d					rpos_lnkCoM[mahru::NO_OF_BODY];		//	CoM position of i-th link w.r.t {B} expressed in {B}
	Eigen::Vector3d					rvel_lnk[mahru::NO_OF_BODY];			//	Velocity of i-th link w.r.t {B} expressed in {B}
	Eigen::Vector3d					rvel_lnkCoM[mahru::NO_OF_BODY];		//	CoM velocity of i-th link w.r.t {B} expressed in {B}

	Eigen::Vector3d					varphi_lnk[mahru::NO_OF_BODY];			//	varphi_lnk = varphi_B + omega_b2lnk_BCS
	Eigen::Vector3d					etadot_lnkCoM[mahru::NO_OF_BODY];		//	varphi_B x r_Gk + rdot_Gk
	Eigen::Vector3d					omega_b2lnk_BCS[mahru::NO_OF_BODY];	//	Angular velocity of i-th link w.r.t {B} exp. in {B}

	Eigen::Matrix3d					I_G_BCS[mahru::NO_OF_BODY];			//	CoM Inertia matrix of i-th link w.r.t. {B}
	Eigen::Matrix3d					Rot_B2Lnk[mahru::NO_OF_BODY];			//	Rotation matrix of i-th link frame w.r.t {B}

	Eigen::Matrix3d					Sk_varphi_lnk[mahru::NO_OF_BODY];		//	Skew symmetric matrix form of varphi_lnk 
	Eigen::Matrix3d					Sk_rpos_lnkCoM[mahru::NO_OF_BODY];		//	Skew symmetric matrix form of rpos_lnkCoM
	Eigen::Matrix3d					Sk_etadot_lnkCoM[mahru::NO_OF_BODY];	//	Skew symmetric matrix form of etadot_lnkCoM

	Eigen::Matrix<double, DOF3, mahru::num_act_joint>		Jp_lnk_BCS[mahru::NO_OF_BODY];			//	Linear Jacobian of i-th link in {B}
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Jr_lnk_BCS[mahru::NO_OF_BODY];			//	Angular Jacobian of i-th link in {B}
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	J_lnkCoM_BCS[mahru::NO_OF_BODY];		//	CoM Jacobian of i-th link in {B}
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Zdotr_lnk[mahru::NO_OF_BODY];			//	Time derivative of Jr_lnk_BCS
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Zdot_lnkCoM[mahru::NO_OF_BODY];		//	Time derivative of J_lnkCoM_BCS


	/////	A rigid-body kinematics variables expressed in {B} : Temporary variables !!!
	Eigen::Vector3d								rpos_A;			//	Position of "A" w.r.t {B} expressed in {B}
	Eigen::Vector3d								rvel_A;			//	Velocity of "A" w.r.t {B} expressed in {B}

	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Jp_BCS;			//	Linear Jacobian of "A" w.r.t {B}
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Jr_BCS;			//	Angular Jacobian of "A" w.r.t {B}
	Eigen::Matrix<double, DOF3, mahru::num_act_joint>		Zdotp_BCS;		//	Time derivative of Jp_BCS
	Eigen::Matrix<double, DOF3, mahru::num_act_joint> 	Zdotr_BCS;		//	Time derivative of Jr_BCS


	//////////////////////////////////////////////////////////////////////
	/////	System coordinates : Expressed in {I}
	//////////////////////////////////////////////////////////////////////
	Eigen::Vector3d								p_B;			//	Position of floating-base body expressed in {I}
	Eigen::Vector4d								quat_B;			//	Quaternion of floating-base body expressed in {I}
	Eigen::Matrix<double, mahru::num_act_joint, 1>		q;				//	Active joint position
	Eigen::Matrix<double, mahru::num_act_joint, 1>		q0;				//	Initial active joint position
	Eigen::Matrix3d								R_B;			//	Rotation matrix of floating-base body expressed in {I}

	Eigen::Vector3d								pdot_B;			//	Linear velocity of floating-base body exp. in {I}
	Eigen::Vector3d								omega_B;		//	Angular velocity of floating-base body exp. in {I} : omega_B = R_B * varphi_B
	Eigen::Vector3d								varphi_B;		//	Angular velocity of floating-base body exp. in {B}
	Eigen::Matrix<double, mahru::num_act_joint, 1>		qdot;			//	Active joint velocity
	Eigen::Matrix3d								Sk_varphi_B;	//	Skew symmetric matrix form of varphi_B

	Eigen::Matrix<double, mahru::nDoF, 1>			xidot;			//	xidot = ( pdot_B, omega_B, qdot ) : Generalized coordinates w.r.t. {I}
	Eigen::Matrix<double, mahru::nDoF, 1>			xiddot;			//	xiddot = ( pddot_B, omegadot_B, qddot )
	Eigen::Matrix<double, mahru::nDoF, 1>			xidot_tmp;		//	(k-1)-th xidot for acceleration computation via. numerical diff.
	Eigen::Matrix<double, mahru::nDoFQuat, 1>	xi_quat;		//	xi_quat = ( p_B, quat_B, q )


	//////////////////////////////////////////////////////////////////////
	/////	Global motion terms : Expressed in {I}
	//////////////////////////////////////////////////////////////////////
	Eigen::Vector3d				p_CoM;		//	CoM position w.r.t. {I}
	Eigen::Vector3d				p_B2CoM;	//	CoM position w.r.t {B} expressed in {I}
	Eigen::Vector3d				pdot_CoM;	//	CoM velocity w.r.t. {I}
	Eigen::Vector3d				pdot_B2CoM;	//	Time derivative of p_B2CoM
	Eigen::Vector3d				pos_ZMP;	//	Zero moment point (ZMP)

	Eigen::Matrix<double, DOF3, mahru::nDoF>	J_CoM;		//	CoM Jacobian matrix of the system
	Eigen::Matrix<double, DOF3, mahru::nDoF>	Jdot_CoM;	//	Time derivative of J_CoM


	//////////////////////////////////////////////////////////////////////
	/////	Dynamics Related Terms : Inertial frame-based
	//////////////////////////////////////////////////////////////////////
	Eigen::Matrix<double, mahru::nDoF, 1>				g_vec;		//	Gravity force vector
	Eigen::Matrix<double, mahru::num_act_joint, 1>			I_actuator;	//	Added joint inertia for geared actuator

	Eigen::Matrix<double, mahru::nDoF, mahru::nDoF>		T_B;
	Eigen::Matrix<double, mahru::nDoF, mahru::nDoF>		M_mat;		//	Joint space inertia matrix
	Eigen::Matrix<double, mahru::nDoF, mahru::nDoF>		C_mat;		//	Coriolis & Centrifugal matrix
	Eigen::Matrix<double, mahru::nDoF, mahru::nDoF>		Mdot_mat;	//	Derivative of joint space inertia matrix

	Eigen::Matrix<double, DOF3, mahru::nDoF>			M_p;		//	Linear part of inertia matrix = J_CoM !!
	Eigen::Matrix<double, DOF3, mahru::nDoF>			M_o;
	Eigen::Matrix<double, DOF3, mahru::nDoF>			C_p;		//	Linear part of Coriolis matrix = Jdot_CoM !!
	Eigen::Matrix<double, DOF3, mahru::nDoF>			C_o;

	/////	Centroidal Momentum Dynamics
	Eigen::Vector3d									l_b, k_b;			//	Linear & angular momentum @{B}
	Eigen::Vector3d 								l_CoM, k_CoM;		//	Linear & angular momentum @CoM
	Eigen::Vector3d 								ldot_CoM, kdot_CoM;	//	Time derivative of linear & angular momentum @ CoM
	Eigen::Matrix<double, DOF6, 1> 					h_b;				//	Momentum w.r.t. {B} expressed in {I}
	Eigen::Matrix<double, DOF6, 1> 					h_CoM;				//	Centroidal momentum expressed in {I}
	Eigen::Matrix<double, DOF6, 1> 					hdot_CoM;			//	Time derivative of h_CoM

	Eigen::Matrix<double, DOF3, mahru::nDoF>			Ap_CoM;		//	Centroidal momentum matrix (CMM) - Linear
	Eigen::Matrix<double, DOF3, mahru::nDoF>			Ar_CoM;		//	Centroidal momentum matrix (CMM) - Angular
	Eigen::Matrix<double, DOF3, mahru::nDoF>			Adotp_CoM;	//	Time derivative of Ap_CoM
	Eigen::Matrix<double, DOF3, mahru::nDoF>			Adotr_CoM;	//	Time derivative of Ar_CoM

	Eigen::Matrix<double, DOF6, mahru::nDoF>			A_CoM;		//	Centroidal momentum matrix (CMM)
	Eigen::Matrix<double, DOF6, mahru::nDoF>			Adot_CoM;	//	Time derivative of CMM

	/////	Kinematic variables for body frame : NOT NECESSARY !!!!
	//vector<Eigen::Matrix<double, DOF3, mahru::nDoF>>	J_lnkCoM;	//	CoM Jacobian of i-th link in {I}
	//vector<Eigen::Matrix<double, DOF3, mahru::nDoF>>	Jdot_lnkCoM;//	Time derivative of Jp_lnkCoM

public:
	CARBML();
	~CARBML() {}

	/////	0. Initialize Robot Model
	void initRobot(const mjModel* model);				//	Class initialization
	void initRobotModel(const mjModel* model);			//	Robot model initialization
	void outputSysInformation(const mjModel* model);	//	Information print


	/////	1. Compute Core Kinematics for other use
	void computeMotionCore();	//	Must be called BEFORE ALL OTHERS !!

	void getPose_BCS();			//	Compute position & rotation of all body/link frame w.r.t {B}
	void getJacob_BCS();		//	Compute Jacobian matrix w.r.t {B}
	void getJacobDeriv_BCS();	//	Compute Time derivative of Jacobian matrix w.r.t {B}


	/////	2. Compute Joint Space Dynamics : Inertia, Coriolis & Centrifugal matrix, gravity vector w.r.t inertial frame
	void computeDynamics();

	void getInertiaMatrix();				//	Compute Joint Inertia Matrix
	void getCoriolisCentrifugalMatrix();	//	Compute Coriolis & Centrifugal matrix
	void getGravityForce();					//	Compute Gravity force
	void computeCentroidalDynamics();		//	Compute centroidal dynamics
	void getInertiaDot();					//	Compute joint inertia derivative matrix


	/////	3. Compute CoM Kinematics of the System : Position, Jacobian and etc.
	void computeCoMKinematics();

	void getCoMPosition();				//	Compute CoM position of the system w.r.t {I}
	void getCoMJacobian();				//	Compute CoM Jacobian of the system w.r.t {I}
	void getCoMJacobianDeriv();			//	Compute time derivative of CoM Jacobian


	/////	4. Compute position & rotation of point 'A' fixed on body 'ID_body' expressed in {I}
	void getBodyPose(const int& ID_body, const Eigen::Vector3d& pos_offset, const Eigen::Matrix3d& Rot_offset, \
					Eigen::Vector3d& pos_body, Eigen::Matrix3d& Rot_body);

	//	Compute pose of a link frame pose
	void getLinkPose(const int& ID_body, Eigen::Vector3d& pos_lnk, Eigen::Matrix3d& Rot_lnk);

	//	Compute pose of a link CoM
	void getLinkCoMPose(const int& ID_body, Eigen::Vector3d& pos_lnkCoM, Eigen::Matrix3d& Rot_lnkCoM);


	/////	5. Compute Jacobian & its time derivative of a point 'A' fixed on body 'ID_body' w.r.t {I}
	void getBodyJacob(const int& ID_body, const Eigen::Vector3d& point_I, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jp_body);
	void getBodyJacob(const int& ID_body, const Eigen::Vector3d& point_I, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jp_body, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jr_body);
	void getBodyJacobDeriv(const int& ID_body, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jdotp_body);
	void getBodyJacobDeriv(const int& ID_body, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jdotp_body, Eigen::Matrix<double, DOF3, mahru::nDoF>& Jdotr_body);


	/////	Misc. functions !
	void clearCapacity();															//	Clear capacity of std::vector()
	void assignCapacity();															//	Assign capacity of std::vector()


	/////	Inline functions !
	inline int&		FlagDynamicsDone() { return IsDynamicsDone; }
	inline int&		BodyID_ActJntStart() { return _actJnt_start_bodyID; }

	inline sysReal& getTotalMass() { return mass_G; }
	inline sysReal&	getSamplingTime() { return _del_T; }
	inline sysReal& getGravityConst() { return _g_const; }
};
