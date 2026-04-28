/*
	This file is written to describe total contact wrench for robots.

	Author : Joonhee Jo
	Department : Cognitive & Collaborative Robotics Research Group
				@ Korea Institute of Science Technology (KIST)

	Date : 2022. 03. 14

	note:

	all variables in the CWC class describes 1 contact wrench/force in local frame.


	lowercase letters = vector or scalar
	uppercase letters = matrix

*/

#pragma once
#include "Contact/FrictionCone.h"
#include <Eigen/Dense>
// #include <iostream>

// contact wrench model
#define CONTACT_WRENCH_MODEL 1	// 1 : for MAHRU-2WL


// contact wrench dof varies according to the contact model.
#if CONTACT_WRENCH_MODEL == 1					// contact wrench for MAHRU-2WL
#define CONTACT_WRENCH_DOF DOF3		// frictional point contact
#define NUM_UPC	CONE_POLYGON1					// unknowns per contact wrench
#endif





// template <unsigned num_m, typename T>
// class CstVector;

// template <unsigned num_m, unsigned num_n, typename T>
// class CstMatrix;

// contact wrench cone for wheel
class CContactWrenchCone
{
private:

	// CFrictionCone	cone;

	double _K_f;											// linear friction coefficient
	double _K_mu;											// angular friction coefficient

public:
	CFrictionCone	cone;

	Eigen::Matrix<double, NUM_UPC, NUM_UPC>	E_cone;

	// friction cone 
	Eigen::Matrix<double, CONTACT_WRENCH_DOF, NUM_UPC> W;				// basis matrix for total friction pyramid
	Eigen::Matrix<double, DOF3, NUM_UPC> Wp;							// basis matrix for total friction pyramid
	Eigen::Matrix<double, NUM_UPC, CONTACT_WRENCH_DOF> W_T;				// basis matrix for total friction pyramid


public:
	CContactWrenchCone() : _K_f(0.0), _K_mu(0.0)
	{
	}
	~CContactWrenchCone() {}

	void init(double K_f, double K_mu);
#if CONTACT_WRENCH_MODEL == 1 // contact wrench for MAHRU-2WL
	void computeContactWrenchCone();
#endif


};

// #include "Contact/ContactWrenchCone.cpp"


