#ifndef __JPOSCTRL_STATE_HPP__
#define __JPOSCTRL_STATE_HPP__

#include <yaml-cpp/yaml.h>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"
#include "Utils/JointTrajectory.h"

using namespace robot_name;

template <typename T>
class FSM_JPosCtrlState : public States {
public:
    explicit FSM_JPosCtrlState(RobotData& robot);
    ~FSM_JPosCtrlState() {};

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {};
    void runTransition() override {};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData*     robot_data_;
    mujoco::TrajVizUtil* viz_ = nullptr;

    void updateModel();
    void updateJPosPlanner();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);

    double                                  t_mov_;
    CP2P_Traj<num_act_joint, T>             Joint_Traj_;
    Eigen::Matrix<T, num_act_joint, 1>      jpos_init_;
    Eigen::Matrix<T, num_act_joint, 1>      jpos_ref_;

    Eigen::Matrix<T, num_act_joint, 1>      target_jpos;
    Eigen::Matrix<T, num_act_joint, 1>      target_jvel;
    Eigen::Matrix<T, num_act_joint, 1>      target_jacc;
};

#endif // __FSM_JPOSCTRL_HPP__
