#pragma once

#include "RobotDefinition.hpp"
#include "RobotStates.hpp"
#include "StateMachine/States.hpp"

#include <vector>

namespace mujoco {
class TrajVizUtil;
}

enum StateList {
    PASSIVE = 0,
    FSM_JPosCtrl = 1,
    FSM_BalanceCtrl,
    NUM_STATE
};

class StateMachineCtrl {
public:
    explicit StateMachineCtrl(RobotData& robot);
    ~StateMachineCtrl();

    StateMachineCtrl(const StateMachineCtrl&) = delete;
    StateMachineCtrl& operator=(const StateMachineCtrl&) = delete;

    void initialize();
    void runState();
    void setVisualizer(mujoco::TrajVizUtil* visualizer);

    std::vector<States*> state_list_;
    States* current_state_ = nullptr;
    States* next_state_ = nullptr;

private:
    bool first_run_ = true;
};
