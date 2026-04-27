#pragma once

namespace mujoco {
class TrajVizUtil;
}

class States {
public:
    virtual ~States() = default;

    double state_time = 0.0;

    virtual void setVisualizer(mujoco::TrajVizUtil*) {}

    virtual void onEnter() = 0;
    virtual void runNominal() = 0;
    virtual void checkTransition() = 0;
    virtual void runTransition() = 0;
};
