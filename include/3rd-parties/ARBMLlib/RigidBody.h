//
//	File name : RigidBody.h
//
//	It describes the information of the rigid body.
//
//	Department : Cognitive & Collaborative Robotics Research Group
//				@ Korea Institute of Science and Technology (KIST)
// 
//	============================================================================
//
//	# NOTE
//
//		Prefix "_" = private members
//		Lowercase letters : vector or scalar
//		Uppercase letters : matrix
//
//	* variables
//		rho : CoM position vector from local frame (const.)
//		pos	: position vector from {0} to {1} w.r.t. {0}
//		Rot : rotation matrix from {0} to {1}
// 
//	* orientation :  3D orientation of the body frame in local coordinates about moving frame. (alpha, beta, gamma)
//		eul_x : x axis orientation
//		eul_y : y axis orientation
//		eul_z : z axis orientation
//
//	============================================================================
// 
//	# Revision History
// 
//		* 2021.07 : Originated by Dr. Joonhee Jo
//		* 2023.03 : Major rev. by Dr Yonghwan Oh
//
#pragma once

#include "Robotics_func.h"

/////	Joint Types : ���� �ٲ��� ���� !!!! (MuJoCo�� mjtJOINT�� ���� ����!)
enum JointType {
	FreeJoint = 0,		//	= 0 : mjJNT_FREE (MUST BE !!)
	BallJoint,			//	= 1 : mjJNT_BALL (MUST BE !!)
	PrismaticJoint,		//	= 2 : mjJNT_SLIDE (MUST BE !!)
	RevoluteJoint,		//	= 3 : mjJNT_HINGE (MUST BE !!)
	UndefinedJoint		//	= 4 : User defined !!
};


enum MotionComp {
	X_AXIS = 0,
	Y_AXIS,
	Z_AXIS
};


class CRigidBody {

private:
	/////////////////////////////////// Link Property //////////////////////////////////////
	
	// Euler angle parameter : eul_x, eul_y,eul_z for ZYX presentation (alpha, beta, gamma) about moving frame.
	// convention order : x1 rotation by alpha. y2 rotation by beta. z3 rotation by gamma. 
	/////////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int	_jointType;
	unsigned int	_jointAxis;
	Eigen::Vector3d	_jointVec;

	sysReal			_mass;		//	mass of link : const
	Eigen::Vector3d	_rho;		//	Const. CoM position vector of i-th link from its local body frame
	Eigen::Matrix3d _I_G;		//	Const. CoM Inertia Matrix@CoM (same orientation as local body frame)

	Eigen::Vector3d	_pos0_link;	//	Const. initial position offset of {i}-frame from {p(i)}-frame
	Eigen::Matrix3d _Rot0_CoM;	//	Const. rotation matrix btw body frame & principal axis frame
	Eigen::Matrix3d	_Rot0_link;	//	Const. initial rotation offset rel. to parent frame

	Eigen::Vector3d	_pos_CoM;	//	CoM position vector i-th link/body frame rel. to p(i) frame
	Eigen::Vector3d	_pos_link;	//	Position vector of i-th body/link frame rel. to p(i) frame
	Eigen::Matrix3d	_Rot_CoM;	//	CoM Rotation(principal axes) i-th body/link frame rel. to p(i) frame
	Eigen::Matrix3d	_Rot_link;	//	Rotation matrix i-th body/link frame rel. to p(i) frame

public:
	CRigidBody();
	CRigidBody(Eigen::Matrix4d& T0);
	~CRigidBody() {}

	void setProperty(const Eigen::Vector3d& pos_I, const Eigen::Vector4d& quat_I, const sysReal mass_I, const Eigen::Vector3d& rho_I,
					const Eigen::Matrix3d& Rot_G, const Eigen::Matrix3d& princial_I_G, const int& j_type);
	void setProperty(const Eigen::Vector3d& pos_I, const Eigen::Matrix3d& Rot_I, const sysReal mass_I, const Eigen::Vector3d& rho_I,
					const Eigen::Matrix3d& Rot_G, const Eigen::Matrix3d& princial_I_G,
					const int& j_type, const Eigen::Vector3d& j_axis, const int& jtype_dir);

	void ElemetaryTransformation(const sysReal& q_I);
	void ElemetaryTransformation(const Eigen::Vector3d& pos_I, const Eigen::Matrix3d& Rot_I);
	void ElemetaryTransformation(const Eigen::Vector3d& pos_I, const Eigen::Matrix3d& Rot_I, const sysReal& q_I);

	inline Eigen::Vector3d		& pos_CoM()					{ return _pos_CoM; }
	inline Eigen::Vector3d		& pos_Link()				{ return _pos_link; }

	inline Eigen::Matrix3d		& Rot_CoM()					{ return _Rot_CoM; }
	inline Eigen::Matrix3d		& Rot_Link()				{ return _Rot_link; }

	inline Eigen::Vector3d		& pos_localCoM()			{ return _rho; }
	inline Eigen::Matrix3d		& Rot_localCoM()			{ return _Rot0_CoM; }
	inline Eigen::Matrix3d		& Inertia_CoM()				{ return _I_G; }

	inline sysReal				& get_mass()				{ return _mass; }
	inline unsigned				& JointAxisType()			{ return _jointType; }
	inline unsigned				& JointAxisDir()			{ return _jointAxis; }
};

