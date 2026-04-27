#include "StateMachine/StateMachineCtrl.hpp"

#include "StateMachine/FSM_JPosCtrl.hpp"

#include <iostream>

StateMachineCtrl::StateMachineCtrl(RobotData& robot)
{
    state_list_.resize(StateList::NUM_STATE, nullptr);
    state_list_[StateList::FSM_JPosCtrl] = new FSM_JPosCtrlState<double>(robot);

    current_state_ = nullptr;
    next_state_ = nullptr;
    std::cout << "[ StateMachineCtrl ] constructed" << std::endl;
}

StateMachineCtrl::~StateMachineCtrl()
{
    for (States* s : state_list_) {
        delete s;
    }
}

void StateMachineCtrl::initialize()
{
    current_state_ = state_list_[StateList::FSM_JPosCtrl];
    next_state_ = current_state_;
    first_run_ = true;
}

void StateMachineCtrl::runState()
{
    if (first_run_) {
        current_state_->onEnter();
        first_run_ = false;
    }
    current_state_->runNominal();
}

void StateMachineCtrl::setVisualizer(mujoco::TrajVizUtil* visualizer)
{
    for (States* state : state_list_) {
        if (state) {
            state->setVisualizer(visualizer);
        }
    }
}
