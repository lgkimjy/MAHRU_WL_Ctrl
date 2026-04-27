#include "FSM_JPosCtrl.hpp"

#include <cmath>
#include <iostream>
#include <string>

template <typename T>
FSM_JPosCtrlState<T>::FSM_JPosCtrlState(RobotData& robot) :
    robot_data_(&robot)
{
    Joint_Traj_.is_moving_ = false;
    std::cout << "[ FSM_JPosCtrlState ] Constructed" << std::endl;
}

template <typename T>
void FSM_JPosCtrlState<T>::onEnter()
{
    std::cout << "[ FSM_JPosCtrlState ] OnEnter" << std::endl;

    if (viz_) {
        viz_->clear();
    }

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_JPosCtrl_config.yaml");
    Joint_Traj_.is_moving_ = false;
    jpos_init_ = robot_data_->fbk.jpos;

    this->state_time = 0.0;
}

template <typename T>
void FSM_JPosCtrlState<T>::runNominal()
{
    updateJPosPlanner();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}

template <typename T>
void FSM_JPosCtrlState<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    viz_ = visualizer;
}

template <typename T>
void FSM_JPosCtrlState<T>::updateCommand()
{
    robot_data_->ctrl.jpos_d = target_jpos;
    robot_data_->ctrl.jvel_d = target_jvel;
    robot_data_->ctrl.torq_d = robot_data_->param.Kp.asDiagonal() * (robot_data_->ctrl.jpos_d - robot_data_->fbk.jpos) + \
                                robot_data_->param.Kd.asDiagonal() * (robot_data_->ctrl.jvel_d - robot_data_->fbk.jvel);
}

template <typename T>
void FSM_JPosCtrlState<T>::updateJPosPlanner()
{
    if(Joint_Traj_.is_moving_ == false) 
    {
        std::cout << "[ FSM_JPosCtrlState ][ JPosCtrl ] Set to initial joint position" << std::endl;
        Joint_Traj_.setTargetPosition(jpos_init_, jpos_ref_, t_mov_, 0.001, QUINTIC);
    }
    Joint_Traj_.computeTraj(target_jpos, target_jvel, target_jacc);
}

template <typename T>
void FSM_JPosCtrlState<T>::updateModel()
{
}

template <typename T>
void FSM_JPosCtrlState<T>::updateVisualization()
{
    if (!viz_) return;

    viz_->sphere("jpos/base",
        robot_data_->fbk.p_B,
        0.035, {0.3f, 0.0f, 0.3f, 0.8f}
    );
    viz_->frame("jpos/base_frame",
        robot_data_->fbk.p_B,
        robot_data_->fbk.R_B,
        0.1, 0.005
    );
    viz_->pushTrail("jpos/base_trail",
        robot_data_->fbk.p_B,
        robot_data_->fbk.R_B,
        500, 4.0, {0.1f, 0.8f, 1.0f, 0.35f},
        100, 0.1, 0.005
    );
}

template <typename T>
void FSM_JPosCtrlState<T>::readConfig(std::string config_file)
{
    std::string path = config_file;
    YAML::Node yaml_node = YAML::LoadFile(path.c_str());
    try {
        for(int i = 0; i < num_act_joint; i++) {
            jpos_ref_[i] = yaml_node["target_jpos"][i].as<T>() * M_PI / 180.0;
            robot_data_->param.Kp[i] = yaml_node["Kp"][i].as<T>();
            robot_data_->param.Kd[i] = yaml_node["Kd"][i].as<T>();
        }
        t_mov_ = yaml_node["t_mov"].as<T>();
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading target_jpos from config file: " << e.what() << std::endl;
    }
}

// template class FSM_JPosCtrlState<float>;
template class FSM_JPosCtrlState<double>;