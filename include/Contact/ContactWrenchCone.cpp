#include "ContactWrenchCone.h"


void CContactWrenchCone::init(double K_f, double K_mu)
{
	cone.initFrictionCone(K_f, K_mu);
	E_cone.setIdentity();

	_K_f = K_f;
}

#if CONTACT_WRENCH_MODEL == 1 // contact wrench for MAHRU-2WL
void CContactWrenchCone::computeContactWrenchCone()
{
	//////////////////////////////////////////////////////////////////////////
	// cone approximation. m-dimensional polygon.
	//////////////////////////////////////////////////////////////////////////
	W.block(0, 0, DOF3, CONE_POLYGON1) = cone.Ubar1;
	W_T = W.transpose();
}
#endif

