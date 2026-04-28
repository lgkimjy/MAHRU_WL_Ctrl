//
//	File name : RigidBody.cpp
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
//	============================================================================
// 
//	# Revision History
// 
//		* 2021.07 : Originated by Dr. Joonhee Jo
//		* 2023.03 : Major rev. by Dr Yonghwan Oh
//
#include "RigidBody.h"


CRigidBody::CRigidBody() : _mass(0.0), _jointType(0), _jointAxis(0)
{
	_jointVec.setZero();

	_rho.setZero();
	_I_G.setZero();
	_pos_CoM.setZero();

	_pos0_link.setZero();
	_Rot0_link.setIdentity();
	_Rot0_CoM.setIdentity();

	_Rot_link.setIdentity();
	_pos_link.setZero();
}



CRigidBody::CRigidBody(Eigen::Matrix4d& HTmat_I) : _mass(0.0), _jointType(0), _jointAxis(0)
{
	_pos_link = HTmat_I.col(3).head(3);
	_Rot_link = HTmat_I.block(0, 0, 3, 3);

	_rho.setZero();
	_I_G.setZero();
	_pos0_link.setZero();
	_pos_CoM.setZero();

	_Rot0_link.setIdentity();
	_Rot0_CoM.setIdentity();
}



///////////////////////////////////////////////////////////////////////////
//	Set Local Link Property : Rotation is given by quaternion !
///////////////////////////////////////////////////////////////////////////
void CRigidBody::setProperty(const Eigen::Vector3d& p_offset, const Eigen::Vector4d& quat_offset, const sysReal mass,
							const Eigen::Vector3d& rho, const Eigen::Matrix3d& R0_CoM, const Eigen::Matrix3d& I_diag,
							const int& joint_type)
{
	_jointType = joint_type;

	_pos0_link = p_offset;
	_Rot0_link = _Quat2Rot(quat_offset);
	_Rot0_CoM = R0_CoM;

	_mass = mass;
	_rho = rho;													//	Local CoM offset from its local body/link frame (LCS)
	_I_G = _Rot0_CoM * I_diag * Eigen::Transpose(_Rot0_CoM);	//	CoM에서 Inertia! : _I_G = _R_G * I_diag * _R_G^T

	_Rot_link = _Rot0_link;
	_pos_link = _pos0_link;

	_Rot_CoM = _Rot0_link * _Rot0_CoM;
	_pos_CoM = _pos0_link + _Rot0_link * _rho;
}



///////////////////////////////////////////////////////////////////////////
//	Set Local Link Property
///////////////////////////////////////////////////////////////////////////
void CRigidBody::setProperty(const Eigen::Vector3d& p_offset, const Eigen::Matrix3d& R_offset, const sysReal mass,
							const Eigen::Vector3d& rho, const Eigen::Matrix3d& R0_CoM, const Eigen::Matrix3d& I_diag,
							const int& jointType_I, const Eigen::Vector3d& jointVec_I, const int& jointAxis_I)
{
	_jointType = jointType_I;
	_jointVec = jointVec_I;
	_jointAxis = jointAxis_I;

	_pos0_link = p_offset;
	_Rot0_link = R_offset;
	_Rot0_CoM = R0_CoM;

	_mass = mass;												//	Link mass
	_rho = rho;													// Local CoM position w.r.t own link/body frame
	_I_G = _Rot0_CoM * I_diag * Eigen::Transpose(_Rot0_CoM);	// CoM Inertia! (CoM frame // Body frame)

	_Rot_link = _Rot0_link;
	_pos_link = _pos0_link;

	_Rot_CoM = _Rot0_link * _Rot0_CoM;
	_pos_CoM = _pos0_link + _Rot0_link * _rho;
}



///////////////////////////////////////////////////////////////////////////
//	Elementary Coordinate Transformation from parent body/link frame
//		- Position vector & Rotation Matrix
///////////////////////////////////////////////////////////////////////////
void CRigidBody::ElemetaryTransformation(const Eigen::Vector3d& pos_I, const Eigen::Matrix3d& Rot_I)
{
	_pos_link = pos_I;
	_Rot_link = Rot_I;

	_pos_CoM = _pos_link + _Rot_link * _rho;
	_Rot_CoM = _Rot_link * _Rot0_CoM;
}



///////////////////////////////////////////////////////////////////////////
//	Elementary Coordinate Transformation from parent body/link frame
//		- Motion Axis : X, Y, Z arbitrary !!
//		- Position vector & Rotation Matrix
///////////////////////////////////////////////////////////////////////////
void CRigidBody::ElemetaryTransformation(const Eigen::Vector3d& pos_I, const Eigen::Matrix3d& Rot_I, const sysReal& q)
{
	sysReal s, c;

	if (_jointType == FreeJoint) {
		_pos_link = pos_I;
		_Rot_link = Rot_I;
	}
	else if (_jointType == PrismaticJoint) {
		_Rot_link = _Rot0_link;
		_pos_link = _pos0_link + _Rot0_link.col(_jointAxis) * q;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == X_AXIS) {
		s = sin(_jointVec(X_AXIS) * q);
		c = cos(_jointVec(X_AXIS) * q);

		_Rot_link <<  _Rot0_link(0, 0),  c * _Rot0_link(0, 1) + s * _Rot0_link(0, 2), -s * _Rot0_link(0, 1) + c * _Rot0_link(0, 2),
					  _Rot0_link(1, 0),  c * _Rot0_link(1, 1) + s * _Rot0_link(1, 2), -s * _Rot0_link(1, 1) + c * _Rot0_link(1, 2),
					  _Rot0_link(2, 0),  c * _Rot0_link(2, 1) + s * _Rot0_link(2, 2), -s * _Rot0_link(2, 1) + c * _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == Y_AXIS) {
		s = sin(_jointVec(Y_AXIS) * q);
		c = cos(_jointVec(Y_AXIS) * q);

		_Rot_link <<  c * _Rot0_link(0, 0) - s * _Rot0_link(0, 2), _Rot0_link(0, 1), s * _Rot0_link(0, 0) + c * _Rot0_link(0, 2),
					  c * _Rot0_link(1, 0) - s * _Rot0_link(1, 2), _Rot0_link(1, 1), s * _Rot0_link(1, 0) + c * _Rot0_link(1, 2),
					  c * _Rot0_link(2, 0) - s * _Rot0_link(2, 2), _Rot0_link(2, 1), s * _Rot0_link(2, 0) + c * _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == Z_AXIS) {
		s = sin(_jointVec(Z_AXIS) * q);
		c = cos(_jointVec(Z_AXIS) * q);

		_Rot_link <<  c * _Rot0_link(0, 0) + s * _Rot0_link(0, 1), -s * _Rot0_link(0, 0) + c * _Rot0_link(0, 1), _Rot0_link(0, 2),
					  c * _Rot0_link(1, 0) + s * _Rot0_link(1, 1), -s * _Rot0_link(1, 0) + c * _Rot0_link(1, 1), _Rot0_link(1, 2),
					  c * _Rot0_link(2, 0) + s * _Rot0_link(2, 1), -s * _Rot0_link(2, 0) + c * _Rot0_link(2, 1), _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == UndefinedJoint) {
		_Rot_link = _Rot0_link;
		_pos_link = _pos0_link;
	}
	else
		_ErrorMsg("ZZZ zzz !!!");

	_pos_CoM = _pos_link + _Rot_link * _rho;
	_Rot_CoM = _Rot_link * _Rot0_CoM;
}



///////////////////////////////////////////////////////////////////////////
//	Elementary Coordinate Transformation from parent body/link frame
//		- Motion Axis : X, Y, Z arbitrary !!
//		- Position vector & Rotation Matrix
///////////////////////////////////////////////////////////////////////////
void CRigidBody::ElemetaryTransformation(const sysReal& q_I)
{
	sysReal s, c;

	if (_jointType == PrismaticJoint) {
		_Rot_link = _Rot0_link;
		_pos_link = _pos0_link + _Rot0_link.col(_jointAxis) * q_I;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == X_AXIS) {
		s = sin(_jointVec(X_AXIS) * q_I);
		c = cos(_jointVec(X_AXIS) * q_I);

		_Rot_link <<  _Rot0_link(0, 0),  c * _Rot0_link(0, 1) + s * _Rot0_link(0, 2), -s * _Rot0_link(0, 1) + c * _Rot0_link(0, 2),
					  _Rot0_link(1, 0),  c * _Rot0_link(1, 1) + s * _Rot0_link(1, 2), -s * _Rot0_link(1, 1) + c * _Rot0_link(1, 2),
					  _Rot0_link(2, 0),  c * _Rot0_link(2, 1) + s * _Rot0_link(2, 2), -s * _Rot0_link(2, 1) + c * _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == Y_AXIS) {
		s = sin(_jointVec(Y_AXIS) * q_I);
		c = cos(_jointVec(Y_AXIS) * q_I);

		_Rot_link <<  c * _Rot0_link(0, 0) - s * _Rot0_link(0, 2), _Rot0_link(0, 1), s * _Rot0_link(0, 0) + c * _Rot0_link(0, 2),
					  c * _Rot0_link(1, 0) - s * _Rot0_link(1, 2), _Rot0_link(1, 1), s * _Rot0_link(1, 0) + c * _Rot0_link(1, 2),
					  c * _Rot0_link(2, 0) - s * _Rot0_link(2, 2), _Rot0_link(2, 1), s * _Rot0_link(2, 0) + c * _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == RevoluteJoint && _jointAxis == Z_AXIS) {
		s = sin(_jointVec(Z_AXIS) * q_I);
		c = cos(_jointVec(Z_AXIS) * q_I);

		_Rot_link <<  c * _Rot0_link(0, 0) + s * _Rot0_link(0, 1), -s * _Rot0_link(0, 0) + c * _Rot0_link(0, 1), _Rot0_link(0, 2),
					  c * _Rot0_link(1, 0) + s * _Rot0_link(1, 1), -s * _Rot0_link(1, 0) + c * _Rot0_link(1, 1), _Rot0_link(1, 2),
					  c * _Rot0_link(2, 0) + s * _Rot0_link(2, 1), -s * _Rot0_link(2, 0) + c * _Rot0_link(2, 1), _Rot0_link(2, 2);

		_pos_link = _pos0_link;
	}
	else if (_jointType == UndefinedJoint) {
		_Rot_link = _Rot0_link;
		_pos_link = _pos0_link;
	}
	else
		_ErrorMsg("ZZZ zzz !!!");

	_Rot_CoM = _Rot_link * _Rot0_CoM;
	_pos_CoM = _pos_link + _Rot_link * _rho;
}

