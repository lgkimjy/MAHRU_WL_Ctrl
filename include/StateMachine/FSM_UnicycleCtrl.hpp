#ifndef __FSM_UNICYCLECTRL_HPP__
#define __FSM_UNICYCLECTRL_HPP__

#include <string>
#include <mujoco/mujoco.h>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

#include "3rd-parties/ARBMLlib/ARBML.h"
#include "Controller/ConvexMPC/ConvexMpc.h"
#include "WBC/WeightedWBC.hpp"

using namespace mahru;

template <typename T>
class FSM_UnicycleCtrlState : public States {
public:
    explicit FSM_UnicycleCtrlState(RobotData& robot);
    ~FSM_UnicycleCtrlState() { delete arbml_; }

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {};
    void runTransition() override {};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData*  robot_data_;
    CARBML*     arbml_ = nullptr;
    mujoco::TrajVizUtil* viz_ = nullptr;
    WeightedWBC weighted_wbc_;

    Eigen::Matrix<T, num_act_joint, 1>      jpos_0_;
    Eigen::Vector3d                         p_CoM_nominal_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         p_CoM_wbc_d_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         pdot_CoM_wbc_d_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                         pddot_CoM_wbc_ff_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d                         R_B_wbc_d_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix<T, num_act_joint, 1>      torq_wbc_ =
        Eigen::Matrix<T, num_act_joint, 1>::Zero();
    Eigen::Matrix<double, ConvexMpc::kForceDim, 1> nominal_grf_ =
        Eigen::Matrix<double, ConvexMpc::kForceDim, 1>::Zero();
    bool                                    is_wbc_solved_ = false;
    bool                                    enable_sway_ = true;
    Eigen::Vector3d                         sway_amplitude_ = Eigen::Vector3d::Zero();
    double                                  sway_frequency_hz_ = 0.1;
    double                                  sway_ramp_time_ = 1.0;

    void updateSwayReference();
    void computeFourContactWBC();
    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);

    //////////////// ARBML ////////////////
    Eigen::Vector3d                         p_lnk[NO_OF_BODY];
    Eigen::Matrix3d                         R_lnk[NO_OF_BODY];

    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jp_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jr_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     J_lnkCoM[NO_OF_BODY];

    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdotp_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdotr_lnk[NO_OF_BODY];
    Eigen::Matrix<T, DOF3, mahru::nDoF>     Jdot_lnkCoM[NO_OF_BODY];

    int                                     no_of_EE;
    vector<int>                             id_body_EE;
    vector<Eigen::Vector3d>                 p0_lnk2EE;
    vector<Eigen::Matrix3d>                 R0_lnk2EE;

    vector<Eigen::Vector3d>                 p_EE;
    vector<Eigen::Vector3d>                 pdot_EE;
    vector<Eigen::Vector3d>                 omega_EE;
    vector<Eigen::Matrix3d>                 R_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jp_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jr_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jdotp_EE;
    vector<Eigen::Matrix<T, DOF3, mahru::nDoF>> Jdotr_EE;

    void initEEParameters(const mjModel* model);
    void computeLinkKinematics();
    void computeEEKinematics(Eigen::Matrix<double, mahru::nDoF, 1>& xidot);
    void computeContactKinematics();
    //////////////// ARBML ////////////////
};

#endif // __FSM_UNICYCLECTRL_HPP__
