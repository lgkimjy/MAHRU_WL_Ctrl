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

using namespace mahru;

class SimulationBridge: public SimulationInterface{
public:
    explicit SimulationBridge(const std::string& scene_file,
                              const std::string& log_dir_override = "",
                              bool headless = false);
    ~SimulationBridge() {}

    std::string log_file_name = "stateData.h5";
    std::string log_dir;
    std::unique_ptr<HDF5Logger> logger;
    void RunHeadless(double duration_s,
                     int log_stride = 1,
                     double stop_com_z = -1.0,
                     double stop_roll_deg = -1.0,
                     double stop_pitch_deg = -1.0,
                     double max_wall_time_s = -1.0,
                     double progress_interval_s = -1.0);
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
