#include "FrictionCone.h"


void CFrictionCone::initFrictionCone(double K_f, double K_mu)
{
	_K_f = K_f;
	_K_mu = K_mu;

	vec_ones_cone.setOnes();
	
	
	E3.setIdentity();

	for (int i = 0; i < CONE_POLYGON1; i++)
	{
		_cosMat(i) = cos(2 * M_PI * i / CONE_POLYGON1);
		_sinMat(i) = sin(2 * M_PI * i / CONE_POLYGON1);
	}

	coneApproximation1();
}

void CFrictionCone::coneApproximation1()
{
	//////////////////////////////////////////////////////////////////////////
	// span form of cone approximation. contact wrench cone with m-dimensional polygon for the surface contact.
	//////////////////////////////////////////////////////////////////////////
	Ubar1.block(GET_X_AXIS, 0, 1, CONE_POLYGON1) = _K_f * _cosMat;
	Ubar1.block(GET_Y_AXIS, 0, 1, CONE_POLYGON1) = _K_f * _sinMat;
	Ubar1.block(GET_Z_AXIS, 0, 1, CONE_POLYGON1) = vec_ones_cone;

}
