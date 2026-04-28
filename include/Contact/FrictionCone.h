/*
	This file is written to describe friction cone approximation for humanoid robot.

	Author : Joonhee Jo
	Department : Cognitive & Collaborative Robotics Research Group
				@ Korea Institute of Science Technology (KIST)

	Date : 2021. 08. 14

	note:

	all variables in the FrictionCone class is described w.r.t. {E}. (End-effector frame or foot frame)

	* cone approximation
		1. 4 cones with m-dimensional polygon.
		2. m-polygon + 3 for moment in foot. (1 cone)
		3. no cone with 3 axes forces in each edge.

	prefix "_" = private members;
	lowercase letters = vector or scalar
	uppercase letters = matrix

	User : Jaewoo An
	Date : 2023. 07. 24

	note :

	Marhwu_Wl


*/

#pragma once
// #include "MathCore.h"
#include <Eigen/Dense>
#include "RobotConfig.h"

#define GET_X_AXIS			0 // 0: x-axis, 1: y-axis, 2:z-axis
#define GET_Y_AXIS			1 // 0: x-axis, 1: y-axis, 2:z-axis
#define GET_Z_AXIS			2 // 0: x-axis, 1: y-axis, 2:z-axis

#define CONE_POLYGON1	6
#define CONE_POLYGON2	1


// template <unsigned num_m, typename T>
// class CstVector;

// template <unsigned num_m, unsigned num_n, typename T>
// class CstMatrix;

class CFrictionCone
{
private:
	double _K_f;											// linear friction coefficient
	double _K_mu;											// angular friction coefficient

	Eigen::Matrix<double, 1,  CONE_POLYGON1> vec_ones_cone;
	Eigen::Matrix<double, DOF3, DOF3> E3;				// 3 x 3 identity matrix


	Eigen::Matrix<double, 1, CONE_POLYGON1> _cosMat, _sinMat;				// hyperbolic function matrix for friction pyramid approximation


public:


public:
	CFrictionCone() : _K_f(0.0), _K_mu(0.0)
	{
	}
	~CFrictionCone() {}

	Eigen::Matrix<double, DOF3, CONE_POLYGON1> Ubar1;			// basis matrix 1 for friction pyramid



	void initFrictionCone(double K_f, double K_mu);

	void coneApproximation1();


};

// #include  "Contact/FrictionCone.cpp"