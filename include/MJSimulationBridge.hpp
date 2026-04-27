#ifndef SIMULATION_BRIDGE_HPP
#define SIMULATION_BRIDGE_HPP

#include <unistd.h>
#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <string>

#include "Interface/MuJoCo/SimulationInterface.hpp"
#include "RobotStates.hpp"
#include "RobotStatesLogger.hpp"
#include "StateMachine/StateMachineCtrl.hpp"

using namespace robot_name;

class SimulationBridge: public SimulationInterface{
public:
    explicit SimulationBridge(const std::string& scene_file);
    ~SimulationBridge() {}

    std::string log_file_name = "stateData.h5";
    std::string log_dir;
    std::unique_ptr<HDF5Logger> logger;
    void LogStates();
 
protected:
    void Initialize() override;
    void UpdateSystemObserver() override;
    void UpdateUserInput() override;
    void UpdateControlCommand() override;
    void UpdateSystemVisualInfo() override;

    RobotData robot_;
    StateMachineCtrl state_machine_;
};

#endif
